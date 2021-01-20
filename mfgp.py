import GPy
import numpy as np
import matplotlib.pyplot as plt
from scipydirect import minimize
from sklearn.metrics import mean_squared_error
from scipy.optimize import fmin
import time
import sys 

import abc

class AbstractGP(metaclass=abc.ABCMeta):

    @abc.abstractmethod
    def predict(self, X_test):
        pass

    @abc.abstractmethod
    def plot(self):
        pass

    @abc.abstractmethod
    def plot_forecast(self, forecast_range):
        pass


class NARGP(AbstractGP):

    def __init__(self, input_dim: int, f_high: callable, adapt_steps: int = 0, f_low: callable = None, 
            lf_X: np.ndarray = None, lf_Y: np.ndarray = None, lf_hf_adapt_ratio: int = 1, lower: float =0., upper: float = 1.):
        '''
        input input_dim:
            dimensionality of the input data
        input f_low:
            closed form of a low-fidelity prediction function, 
            if not provided, call self.lf_fit() to train a low-fidelity GP which will be used for low-fidelity predictions instead
        '''
        self.input_dim = input_dim
        self.__f_high_real = f_high
        self.f_low = f_low
        self.adapt_steps = adapt_steps
        self.lf_hf_adapt_ratio = lf_hf_adapt_ratio
        self.lower, self.upper = lower, upper 
        self.bounds = np.zeros((self.input_dim, 2), dtype=np.float)
        self.bounds[:,0], self.bounds[:,1] = lower, upper 
        # self.augm_iterator = EvenAugmentation(self.n, dim=input_dim)
        self.acquired_X = []
        self.error = []
        lf_model_params_are_valid = (f_low is not None) ^ (
            (lf_X is not None) and (lf_Y is not None) and (lf_hf_adapt_ratio is not None))
        assert lf_model_params_are_valid, 'define low-fidelity model either by mean function or by Data'

        self.data_driven_lf_approach = f_low is None

        if self.data_driven_lf_approach:
            self.lf_X = lf_X
            self.lf_Y = lf_Y

            self.lf_model = GPy.models.GPRegression(
                X=lf_X, Y=lf_Y, initialize=True
            )
            self.lf_model.optimize()
            self.__adapt_lf() # TODO: look at this later
            self.__lf_mean_predict = lambda t: self.lf_model.predict(t)[0]
        else:
            self.__lf_mean_predict = f_low

    def fit(self, hf_X):
        self.hf_X = hf_X
        if self.hf_X.ndim == 1:
            self.hf_X = hf_X.reshape(-1,1)
        assert self.hf_X.shape[1] == self.input_dim
        self.hf_Y = self.__f_high_real(self.hf_X)
        assert self.hf_Y.shape == (len(self.hf_X), 1)
        augmented_hf_X = self.__augment_Data(self.hf_X)

        self.hf_model = GPy.models.GPRegression(
            X=augmented_hf_X,
            Y=self.hf_Y,
            kernel=self.NARGP_kernel(),
            initialize=True
        )
        # Following are the ARD steps
        self.hf_model[".*Gaussian_noise"] = self.hf_model.Y.var()*0.01
        self.hf_model[".*Gaussian_noise"].fix()
        self.hf_model.optimize(max_iters = 500)
        self.hf_model[".*Gaussian_noise"].unfix()
        self.hf_model[".*Gaussian_noise"].constrain_positive()
        self.hf_model.optimize_restarts(20, optimizer = "bfgs",  max_iters = 1000, verbose=False)

    def adapt(self, a=None, b=None, plot=None, X_test=None, Y_test=None, verbose=True):
        assert self.adapt_steps > 0
        if plot == 'uncertainty':
            assert self.input_dim == 1
            self.__adapt_plot_uncertainties(
                X_test=X_test, Y_test=Y_test, verbose=verbose)
        elif plot == 'mean':
            assert self.input_dim == 1
            self.__adapt_plot_means(
                X_test=X_test, Y_test=Y_test, verbose=verbose)
        elif plot is None:
            assert self.input_dim > 0
            self.__adapt_no_plot(verbose=verbose, X_test=X_test, Y_test=Y_test)
        else:
            raise Exception(
                'invalid plot mode, use mean, uncertainty or False')

    def __adapt_plot_uncertainties(self, X_test=None, Y_test=None, verbose=False):
        X = np.linspace(self.a, self.b, 200).reshape(-1, 1)
        # prepare subplotting
        subplots_per_row = int(np.ceil(np.sqrt(self.adapt_steps)))
        subplots_per_column = int(np.ceil(self.adapt_steps / subplots_per_row))
        fig, axs = plt.subplots(
            subplots_per_row,
            subplots_per_column,
            sharey='row',
            sharex=True,
            figsize=(20, 10))
        fig.suptitle(
            'Uncertainty development during the adaptation process')
        log_mses = []

        # axs_flat = axs.flatten()
        for i in range(self.adapt_steps):
            acquired_x, min_val = self.get_input_with_highest_uncertainty()
            if verbose:
                print("Step number: {}".format(i))
                print('new x acquired: {} with value: {}'.format(acquired_x, min_val))
            _, uncertainties = self.predict(X)
            # todo: for steps = 1, flatten() will fail
            ax = axs.flatten()[i]
            ax.axes.xaxis.set_visible(False)
            log_mse = self.assess_log_mse(X_test, Y_test)
            log_mses.append(log_mse)
            ax.set_title(
                'log mse: {}, high-f. points: {}'.format(log_mse, len(self.hf_X)))
            ax.plot(X, uncertainties)
            ax.plot(acquired_x.reshape(-1, 1), 0, 'rx')
            self.fit(np.append(self.hf_X, acquired_x))

        # plot log_mse development during adapt process
        plt.figure(2)
        plt.title('logarithmic mean square error')
        plt.xlabel('hf points')
        plt.ylabel('log mse')
        hf_X_len_before = len(self.hf_X) - self.adapt_steps
        hf_X_len_now = len(self.hf_X)
        plt.plot(
            np.arange(hf_X_len_before, hf_X_len_now),
            np.array(log_mses)
        )

    def __adapt_plot_means(self, X_test=None, Y_test=None, verbose=False):
        X = np.linspace(self.a, self.b, 200).reshape(-1, 1)
        for i in range(self.adapt_steps):
            acquired_x, min_val = self.get_input_with_highest_uncertainty()
            if verbose:
                print("Step number: {}".format(i))
                print('new x acquired: {} with value: {}'.format(acquired_x, min_val))
            means, _ = self.predict(X)
            plt.plot(X, means, label='step {}'.format(i))
            self.fit(np.append(self.hf_X, acquired_x))
        plt.legend()

    def __adapt_no_plot(self, verbose=True, eps=1e-6, X_test=None, Y_test=None):
        for i in range(self.adapt_steps):
            t_start = time.time()
            acquired_x, min_val = self.get_input_with_highest_uncertainty()
            t_end = time.time()
            print("time taken to optimize: {}".format(t_end-t_start))
            if verbose:
                print("Step number: {}".format(i))
                print('new x acquired: {} with value: {}'.format(acquired_x, min_val))
            if np.abs(min_val) < eps:
                print("Exiting adaption step because of low uncertainity")
                break
            new_hf_X = np.append(self.hf_X, [acquired_x], axis=0)
            assert new_hf_X.shape == (len(self.hf_X) + 1, self.input_dim)
            t_start = time.time()
            self.fit(new_hf_X)
            t_end = time.time()
            print("time taken to fit : {}".format(t_end-t_start))
            if (X_test is not None) and (Y_test is not None):
                error = self.assess_mse(X_test, Y_test)
                self.error.append(error)

    def __acquisition_curve(self, x):
        if x.ndim == 1:
            X = x[None, :]
        _, uncertainty = self.predict(X)
        return (-1.0* uncertainty)

    def plot_error(self):
        assert len(self.error) > 0, "Size of error list must be greater than 1"
        n = len(self.error) + 1
        x = range(1, n)
        plt.plot(x, self.error)
        plt.yscale('log')
        plt.show()

    def get_input_with_highest_uncertainty(self, restarts: int = 20):
        res = minimize(self.__acquisition_curve, self.bounds)
        return res['x'], res['fun']

    def __adapt_lf(self):
        X = np.linspace(self.a, self.b, 100).reshape(-1, 1)
        for i in range(self.adapt_steps * self.lf_hf_adapt_ratio):
            uncertainties = self.lf_model.predict(X)[1]
            maxIndex = np.argmax(uncertainties)
            new_x = X[maxIndex].reshape(-1, 1)
            new_y = self.lf_model.predict(new_x)[0]

            self.lf_X = np.append(self.lf_X, new_x, axis=0)
            self.lf_Y = np.append(self.lf_Y, new_y, axis=0)

            self.lf_model = GPy.models.GPRegression(
                self.lf_X, self.lf_Y, initialize=True
            )
            self.lf_model.optimize_restarts(
                num_restarts=5,
                optimizer='tnc'
            )

    def predict(self, X_test):
        assert X_test.ndim == 2, "Please input at least a two dimensional array"
        assert X_test.shape[1] == self.input_dim, "Input dimensions are incorrect"
        temp = self.__augment_Data(X_test)
        return self.hf_model.predict(temp)

    def predict_means(self, X_test): # this function is actually not needed
        means, _ = self.predict(X_test) 
        return means

    def predict_variance(self, X_test): # this function is actually not needed
        _, uncertainties = self.predict(X_test)
        return uncertainties

    def plot(self):
        assert self.input_dim == 1, 'data must be 2 dimensional in order to be plotted'
        self.__plot()

    def plot_forecast(self, forecast_range=.5):
        self.__plot(exceed_range_by=forecast_range)

    def assess_mse(self, X_test, y_test):
        assert X_test.shape[1] == self.input_dim
        assert y_test.shape[1] == 1
        predictions = self.predict_means(X_test)
        mse = mean_squared_error(y_true=y_test, y_pred=predictions)
        return mse

    def assess_log_mse(self, X_test, y_test):
        assert X_test.shape[1] == self.input_dim
        assert y_test.shape[1] == 1
        predictions = self.predict_means(X_test)
        mse = mean_squared_error(y_true=y_test, y_pred=predictions)
        log_mse = np.log2(mse)
        return np.round(log_mse, 6)

    def NARGP_kernel(self, kern_class1=GPy.kern.RBF, kern_class2=GPy.kern.RBF, kern_class3=GPy.kern.RBF):
        aug_input_dim = 1
        std_indices = np.arange(aug_input_dim, self.input_dim+1)
        aug_indices = np.arange(aug_input_dim)

        kern1 = kern_class1(aug_input_dim, active_dims=aug_indices)
        kern2 = kern_class2(self.input_dim, active_dims=std_indices)
        kern3 = kern_class3(self.input_dim, active_dims=std_indices)
        return kern1 * kern2 + kern3

    def __plot(self, confidence_inteval_width=2, plot_lf=True, plot_hf=True, plot_pred=True, exceed_range_by=0):
        point_density = 500
        X = np.linspace(self.lower, self.upper * (1 + exceed_range_by),
                        int(point_density * (1 + exceed_range_by))).reshape(-1, 1)
        pred_mean, pred_variance = self.predict(X.reshape(-1, 1))
        pred_mean = pred_mean.flatten()
        pred_variance = pred_variance.flatten()

        if (not self.data_driven_lf_approach):
            self.lf_X = np.linspace(self.lower, self.upper, 50).reshape(-1, 1)
            self.lf_Y = self.__lf_mean_predict(self.lf_X)

        lf_color, hf_color, pred_color = 'r', 'b', 'g'

        plt.figure(3)
        if plot_lf:
            # plot low fidelity
            # plt.plot(self.lf_X, self.lf_Y, lf_color +
                     # 'x', label='low-fidelity')
            plt.plot(X, self.__lf_mean_predict(X), lf_color,
                     label='f_low', linestyle='dashed')

        if plot_hf:
            # plot high fidelity
            plt.plot(self.hf_X, self.hf_Y, hf_color +
                     'x', label='high-fidelity')
            plt.plot(X, self.__f_high_real(X), hf_color,
                     label='f_high', linestyle='dashed')

        if plot_pred:
            # plot prediction
            plt.plot(X, pred_mean, pred_color, label='prediction')
            plt.fill_between(X.flatten(),
                             y1=pred_mean - confidence_inteval_width * pred_variance,
                             y2=pred_mean + confidence_inteval_width * pred_variance,
                             color=(0, 1, 0, .75)
                             )

        plt.legend()

    def __augment_Data(self, X):
        assert X.shape == (len(X), self.input_dim)
        augmented_X = np.zeros((len(X), self.input_dim + 1))
        augmented_X[:, 0] = self.__lf_mean_predict(np.atleast_2d(X))[:,0]
        augmented_X[:, 1:] = X
        return augmented_X

    def __update_input_borders(self, X: np.ndarray):
        if self.a is None and self.b is None:
            self.a = np.min(X, axis=0)
            self.b = np.max(X, axis=0)
        else:
            self.a = np.min([self.a, np.min(X, axis=0)], axis=0)
            self.b = np.max([self.b, np.max(X, axis=0)], axis=0)
