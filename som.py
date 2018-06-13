from scipy.spatial.distance import cdist, pdist
import numpy as np
import sys
import matplotlib.pyplot as plt

class SOMSolver:


    def __init__(self, data, y_dim, train_it=10000, params=(4.,4.,1.,4.)):

        self.train_it = train_it
        self.y_dim = y_dim

        if params is not None:
            self.sigma_0, self.tau_sigma, self.eta_0, self.tau_eta = params
        else:
            self.sigma_0, self.tau_sigma, self.eta_0, self.tau_eta = self.compute_params()

        self.m = y_dim[0]*y_dim[1]
        
        self.x_train = data
        self.n = len(self.x_train)

        self.weights = []
        self.som_train()


    def som_train(self):
        self.weights = self.initialize_weights()
        
        w = np.reshape(self.weights, (self.y_dim[1],self.y_dim[0],3))

        total_iterations = self.train_it
        counter = 0
        for t in range(self.train_it):
            counter += 1
            sys.stdout.write("\r Complete %.1f %%, %s total. %s"%(float((100.0*counter)/total_iterations), counter, total_iterations)) 
            sys.stdout.flush()

            k = np.random.choice(range(self.n))
            x = self.x_train[k]

            dists_arr = cdist([x], self.weights)[0]
            min_ind = np.argmin(dists_arr)

            
            for j in range(len(self.weights)):
                ji = (j, min_ind)
                self.weights[j] = self.weights[j] + self.delta_w_fact(ji, t)*(x-self.weights[j])
            
        w = np.reshape(self.weights, (self.y_dim[1],self.y_dim[0],3))
        plot_som(w, 'fin.png')


    def compute_loss(self):
        dists = []
        tops = []
        for x in self.x_train:
            dist_arr = cdist([x], self.weights)[0]
            min_ind = np.argmin(dist_arr)
            
            tops.append( self.weights[min_ind] )
            dists.append( dist_arr[min_ind] )
    

        return np.sum(dists)


    def initialize_weights(self):

        w = []
        for j in range(self.m):

            rand_arr = np.random.sample(size=self.n)

            rand_arr = [rand_arr/np.sum(rand_arr)]

            arr = self.x_train

            w.append( np.dot(rand_arr, arr)[0] )

        return np.array(w)


    def compute_params(self):

        sigma_0 = max(self.y_dim[0], self.y_dim[1])*.5
        tau_sigma = self.train_it/np.log(sigma_0)
        eta_0 = .08
        tau_eta = self.train_it

        result = sigma_0, tau_sigma, eta_0, tau_eta

        return result


    def grid_from_ind(self, k):
        assert k < self.m, "Oju llargada"

        j = int(k)/self.y_dim[0]
        i = k - j*self.y_dim[0]
        return i, j


    def ind_from_grid(self, i, j):
        assert j < self.y_dim[1] and i < self.y_dim[0], "Oju dimensions"
        return j*self.y_dim[0]+i


    def delta_w_fact(self, ji, t):
        return self.eta(t)*self.top_neigh(ji, t)


    def top_neigh(self, ji, t):        
        return np.exp(-self.S_dist_2(*ji)/(2.*self.sigma(t)))


    def S_dist_2(self, j, i):
        i1, i2 = self.grid_from_ind(i)
        j1, j2 = self.grid_from_ind(j)
        return (j1-i1)**2 + (j2-i2)**2


    def sigma(self, t):
        return self.sigma_0*np.exp(-t/self.tau_sigma)


    def eta(self, t):
        return self.eta_0*np.exp(-t/self.tau_eta)


def plot_som(data, filename):
    fig = plt.figure(figsize=(10, 10))
    ax = fig.add_subplot(111)

    interpolation = 'nearest'#'bilinear'
    plt.imshow(data, interpolation=interpolation)

    plt.title('')
    plt.xlabel('')
    plt.ylabel('')
    ax.get_xaxis().set_visible(False)
    ax.get_yaxis().set_visible(False)

    fig.savefig(filename, format='png')

    plt.close()


if __name__ == '__main__':
    # Example

    colors = np.array(
         [[0., 0., 0.],
          [0., 0., 1.],
          [0., 0., 0.5],
          [0.125, 0.529, 1.0],
          [0.33, 0.4, 0.67],
          [0.6, 0.5, 1.0],
          [0., 1., 0.],
          [1., 0., 0.],
          [0., 1., 1.],
          [1., 0., 1.],
          [1., 1., 0.],
          [1., 1., 1.],
          [.33, .33, .33],
          [.66, .66, .66]
          ])

    input_c = np.reshape(colors, (1,14,3))

    plot_som(input_c, 'input.png')

    var = SOMSolver(colors, y_dim=(70,70), train_it=70000, params=None)

