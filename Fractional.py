import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.cm as cm
from scipy.integrate import cumulative_trapezoid as cumtrapz

class FractionalDominance(object):
    
    def __init__(
    self, 
    max_dominance_param = 10, 
    fractional_param = None, 
    confidence_interval = None,
    dominance_param = None
    ):
        
        self.max_dominance_param = max_dominance_param
        self.dominance_param = dominance_param
        self.fractional_param = fractional_param
        self.confidence_interval = confidence_interval

    def fit(self, X, y, weight = None, dominance_param = True, fractional_param = None):
        
        """
        X = vector of consumption (dataframe)
        y = vector of incomes (dataframe)
        fractional_param = value of fractional dominance in (0,1)
        dominance_param = int (1 for order 1, etc.)
        """
        if fractional_param is None:
            assert dominance_param >= 2, "dominance_param must be greater than 1 when fractional_param is None"

        self.fractional_param = fractional_param
        self.dominance_param = int(dominance_param)
        self.df = X 
        X = X.to_numpy()
        y = y.to_numpy()
        y = y.reshape(-1, 1)
        self.X = X
 
        if weight is None:
            X_all = np.hstack((y, X))
            X_all = X_all[X_all[:,0].argsort()]  
            y = X_all[:,0]
            self.y = y
            n,_ = X.shape
            self.n = n
            x = X_all[:,1::]                # consumption
            x = x.astype(float)
            self.x = x
            t1 = np.ones((n,1))*(1/n)
            t1 = np.cumsum(t1)
            self.t1 = t1
            a = x / x.mean(axis=0, keepdims=True)
            self.a = a
        else:
            weight = np.array(weight).reshape(-1, 1)
            X_all = np.hstack((y, weight, X))
            X_all = X_all[X_all[:,0].argsort()]  
            y = X_all[:,0]
            self.y = y
            n,_ = X.shape
            self.n = n
            x = X_all[:,2::]                # consumption
            x = x.astype(float)
            self.x = x
            weight_order = X_all[:,1].reshape(-1, 1)
            weight_order = weight_order / np.sum(weight_order)
            self.t1 = np.cumsum(weight_order)
            t1 = self.t1
            mean = np.sum(x * weight_order, axis=0) 
            a = np.cumsum(x * weight_order / mean, axis=0)
            self.a = a
            
        C = np.zeros((x.shape[1], n, self.dominance_param))   # CD-CURVES of ORDER 2,3,...
        C[:,:,0] = self.a.T  
        for col in range(x.shape[1]):
            for j in range(1, self.dominance_param):
                C[col,:,j] = cumtrapz(C[col,:,j-1], x=t1.squeeze(), initial=0) 
        self.C = C[:,:,self.dominance_param-2]
        self.C_all_order = C
        
        if fractional_param is not None:
            if self.fractional_param > 0.5:
                h = np.exp((1 / self.fractional_param - 1) * self.t1.squeeze())
            else:
                h = self.t1.squeeze()**(1 / self.fractional_param)
            #h = 1 / (1 + np.exp((1 / self.fractional_param - 1) * self.t1.squeeze()))
            
            CC = np.zeros((x.shape[1],n,self.dominance_param)) # CD-CURVES with interpolations
            CC[col,0,0] = self.a[0,col]*h[0]/2
            for col in range(x.shape[1]):
                for j in range(1,self.dominance_param):          
                    CC[col,0,j] = C[col,0,j-1]*h[0]/2
            for col in range(x.shape[1]):
                for i in range(1,n):
                    CC[col,i,0] = CC[col,i-1,0] + ((h[i] - h[i-1]))*(a[i,col] + a[i-1,col])/2  # order 2 
                    for j in range(1,self.dominance_param): 
                        CC[col,i,j] = CC[col,i-1,j] + ((h[i] - h[i-1]))*(C[col,i,j-1] + C[col,i-1,j-1])/2 # order 3
            self.CC = CC[:,:,self.dominance_param-2]
        
        if fractional_param is not None:
            return CC[:,:,self.dominance_param-2]
        else:
            return C[:,:,self.dominance_param-2]

    def minimal_frac_dominance_all(self):
        '''
        Compute fractional dominance for each and every pairs of commodities 
        Take the dominance parameter and moves to higher order with increment of 0.01 until dominance
        
        '''
        n, k = self.x.shape
        column_names = self.df.columns
        results_bis = np.zeros((self.X.shape[1], self.X.shape[1]))
        frac = 0
        for fractional_param in np.arange(0.1, 1, 0.1):
            if fractional_param > 0.5:
                h = np.exp((1 / fractional_param - 1) * self.t1.squeeze())
            else:
                h = self.t1.squeeze()**(1 / fractional_param)
            CC = np.zeros((k, n))
            for col in range(k):
                CC[col, 0] = self.C_all_order[col, 0, self.dominance_param-1] * h[0] / 2
                for i in range(1, n):
                    CC[col, i] = CC[col, i-1] + ((h[i] - h[i-1]) * (self.C_all_order[col, i, self.dominance_param-2] + self.C_all_order[col, i-1, self.dominance_param-2])) / 2
            
            for j in range (self.X.shape[1]):
                for i in range(j):
                    diff = CC[j,:] - CC[i,:]
                    if frac == 0:
                        if np.all(diff[100:-100] >= 0):
                            results_bis[j,i] = self.dominance_param + fractional_param
                        if np.all(diff[100:-100] <= 0):
                            results_bis[j,i] = self.dominance_param + fractional_param
                    else:    
                        if np.all(diff[100:-100] >= 0) and results_bis[j,i] == 0.0:
                                results_bis[j,i] = self.dominance_param + fractional_param
                        if np.all(diff[100:-100] <= 0) and results_bis[j,i] == 0.0:
                                results_bis[j,i] = self.dominance_param + fractional_param
            frac = frac + 1
        results_bis = np.where(results_bis == None, "", results_bis)
        results_df = pd.DataFrame(results_bis, columns=column_names, index=column_names)

        #results_df.replace(0.0, "cross", inplace=True)
        return display(results_df)

    def graph(self, good_1, good_2):
        '''
        good_1 and good_2 = string (columns names of the dataframe)         
        '''
        position_good_1 = self.df.columns.get_loc(good_1)
        position_good_2 = self.df.columns.get_loc(good_2)
        
        ## PLOT ##
        if self.fractional_param:
            plt.plot(self.t1, self.CC[position_good_1,:], 'b', label=good_1)
            plt.plot(self.t1, self.CC[position_good_2,:], 'r', label=good_2)
            plt.ylabel('Cumulative consumption')
            plt.xlabel('Income percentiles')
            plt.title(f's-Curves of order {self.dominance_param}+{self.fractional_param}')
            plt.legend(loc='best')
            plt.show()            
        else:    
            plt.plot(self.t1, self.C[position_good_1,:], 'b', label=good_1)
            plt.plot(self.t1, self.C[position_good_2,:], 'r', label=good_2)
            plt.ylabel('Cumulative consumption')
            plt.xlabel('Income percentiles')
            plt.title(f's-Curves of order {self.dominance_param}')
            plt.legend(loc='best')
            plt.show()
    
    def graph_all(self):
        """"
        Graph of all s-curves of the dataset (related to the order c+s in the fit function)
        """
        ## PLOT ##
        if self.fractional_param:
            colormap = cm.get_cmap('rainbow', self.CC.shape[0])
            for i in range(self.CC.shape[0]):
                plt.plot(self.t1, self.CC[i, :], color=colormap(i), label=self.df.columns[i])
            plt.ylabel('Cumulative consumption')
            plt.xlabel('Income percentiles')
            plt.title(f's-Curves of order {self.dominance_param}+{self.fractional_param}')
            plt.legend(loc='best')
            plt.show()            
        else:    
            colormap = cm.get_cmap('rainbow', self.C.shape[0])
            for i in range(self.C.shape[0]):
                plt.plot(self.t1, self.C[i, :], color=colormap(i), label=self.df.columns[i])
            plt.ylabel('Cumulative consumption')
            plt.xlabel('Income percentiles')
            plt.title(f's-Curves of order {self.dominance_param}')
            plt.legend(loc='best')
            plt.savefig('high_resolution_plot.png', dpi=500)
            plt.show()
            
    
    def bootstrap(self, good_1, good_2,
                             B=1000,
                             confidence_interval=0.9):

        alpha = (1 - confidence_interval) / 2
        n = self.n

        pos1 = self.df.columns.get_loc(good_1)
        pos2 = self.df.columns.get_loc(good_2)

        curve1 = self.CC[pos1, :] if self.fractional_param else self.C[pos1, :]
        curve2 = self.CC[pos2, :] if self.fractional_param else self.C[pos2, :]

        boot1 = np.zeros((B, n))
        boot2 = np.zeros((B, n))

        for b in range(B):
            idx = np.random.choice(n, size=n, replace=True)

            # Sort to preserve monotonicity
            boot1[b, :] = np.sort(curve1[idx])
            boot2[b, :] = np.sort(curve2[idx])

        lower1 = np.quantile(boot1, alpha, axis=0)
        upper1 = np.quantile(boot1, 1 - alpha, axis=0)

        lower2 = np.quantile(boot2, alpha, axis=0)
        upper2 = np.quantile(boot2, 1 - alpha, axis=0)

        # ---- Plot ----
        #plt.figure(figsize=(9, 6))

        # True curves
        plt.plot(self.t1.flatten(), curve1, label=f"{good_1}")
        plt.plot(self.t1.flatten(), curve2, label=f"{good_2}")

        # Bands
        plt.fill_between(self.t1.flatten(), lower1, upper1, alpha=0.25)
        plt.fill_between(self.t1.flatten(), lower2, upper2, alpha=0.25)

        plt.xlabel("Income percentiles")
        plt.ylabel("Cumulative consumption")
        plt.title(f"Bootstrap CI ({confidence_interval*100:.0f}%)")
        plt.legend()
        plt.show()

        return (lower1, upper1), (lower2, upper2)

    
    def efficiency_gain(self, good_1, good_2):
        position_good_1 = self.df.columns.get_loc(good_1)
        position_good_2 = self.df.columns.get_loc(good_2)
        if self.fractional_param:
            curve1 = self.CC[position_good_1,:]
            curve2 = self.CC[position_good_2,:]
        else:
            curve1 = self.C[position_good_1,:]
            curve2 = self.C[position_good_2,:]
        
        # Eff. gain delta:
        delta = np.zeros((self.n,1))
        for i in range (self.n):
            if curve1[i] != 0 and curve2[i] != 0:
                delta[i] = curve1[i] / (curve2[i])
        delta2 = np.delete(delta, np.where(delta == 0))

        # Crossing points
        differences = curve1 - curve2
        sign_changes = np.where(np.diff(np.sign(differences)))[0]
        intersection_points = []
        income_values_at_intersections = []
        for index in sign_changes:
            # Linear interpolation :
            x1, x2 = self.t1[index], self.t1[index + 1]
            y1, y2 = differences[index], differences[index + 1]
            intersection_x = x1 - y1 * (x2 - x1) / (y2 - y1)
            intersection_points.append(intersection_x)
            
            # Find the income value corresponding to the intersection point
            income_value = np.interp(intersection_x, self.t1, self.y)
            income_values_at_intersections.append(income_value)

        if not intersection_points:
            print("No intersection points found")
        else:
            for i, point in enumerate(intersection_points):
                print(f'Intersection point {i+1}: p={point}, Income={income_values_at_intersections[i]}')
        
        print("Efficiency gain=", np.amin(delta2))
        print("Efficiency gain=", delta2)

        # Plot the curves
        plt.plot(self.t1, curve1, 'r', label=good_1)
        plt.plot(self.t1, curve2, 'g', label=good_2)
        plt.plot(self.t1, delta, 'b', label='Benefits')
        plt.ylabel('Cumulative distribution')
        plt.xlabel('Incomes')
        plt.title('CD-Curves')
        plt.legend(loc='best')
        plt.show()
        

    def convergence(self, good_1, good_2):
        position_good_1 = self.df.columns.get_loc(good_1)
        position_good_2 = self.df.columns.get_loc(good_2)
        if self.fractional_param:
            curve1 = self.CC[position_good_1,:]
            curve2 = self.CC[position_good_2,:]
        else:
            curve1 = self.C[position_good_1,:]
            curve2 = self.C[position_good_2,:]
        
        differences = curve1 - curve2

        # Plot the curves
        plt.plot(self.t1, differences, 'r', label="curve 1-2")
        plt.ylabel('Cumulative distribution')
        plt.xlabel('Incomes')
        plt.title('CD-Curves')
        plt.legend(loc='best')
        plt.show()

    
    def critical_ratios_poverty(self, gamma=None):
        '''
        Compute CD_i / CD_j to find the minimal value of gamma_ij at the poverty line
        '''
        np.seterr(divide='ignore', invalid='ignore')
        column_names = self.df.columns
        results = np.full((self.X.shape[1], self.X.shape[1]), np.nan, dtype=object)
        results_gamma = np.full((self.X.shape[1], self.X.shape[1]), np.nan, dtype=object)
        results_cross = np.full((self.X.shape[1], self.X.shape[1]), np.nan, dtype=object)

        # Poverty line
        median_idx = np.searchsorted(self.t1, 0.5, side='left')
        poverty_line = 0.5 * self.y[median_idx]
        poverty_idx = np.searchsorted(self.y, poverty_line, side='left')
        self.rank_poverty = self.t1[poverty_idx]
        
        for j in range(self.X.shape[1]):
            for i in range(j):
                curve_j = self.CC[j, :] if self.fractional_param else self.C[j, :]
                curve_i = self.CC[i, :] if self.fractional_param else self.C[i, :]

                sliced_curve_j = curve_j[100:-100]
                sliced_curve_i = curve_i[100:-100]
                diff = sliced_curve_j - sliced_curve_i
                ratio = sliced_curve_j / sliced_curve_i
                delta = ratio[poverty_idx - 100]
                 
                if np.all(diff >= 0):
                    results[j, i] = delta
                    results[i, j] = delta
                    if gamma:
                        diffs = ratio - gamma
                        nonnegative_indices = np.where(diffs >= 0)[0]
                        if nonnegative_indices.size > 0:
                            closest_index = np.argmin(diffs[nonnegative_indices])
                            actual_index = nonnegative_indices[closest_index]
                            percentile = (actual_index - 100) / len(self.y)
                            results_gamma[i, j] = percentile
                            results_gamma[j, i] = percentile

                elif np.all(diff <= 0):
                    results[j, i] = 1 / delta if delta != 0 else "inf"
                    results[i, j] = 1 / delta if delta != 0 else "inf"
                    if gamma:
                        diffs = 1 / ratio - gamma
                        nonnegative_indices = np.where(diffs >= 0)[0]
                        if nonnegative_indices.size > 0:
                            closest_index = np.argmin(diffs[nonnegative_indices])
                            actual_index = nonnegative_indices[closest_index]
                            percentile = (actual_index - 100) / len(self.y)
                            results_gamma[i, j] = percentile
                            results_gamma[j, i] = percentile

                else:
                    cutoff = poverty_idx - 100
                    if cutoff > 1:
                        diff_before_poverty = diff[:cutoff]
                        diff_before_poverty = diff_before_poverty[~np.isnan(diff_before_poverty)]
                        sign_changes = np.where(np.diff(np.sign(diff_before_poverty)) != 0)[0]
                        if sign_changes.size > 0:
                            percentile = sign_changes[0] / len(self.y)
                            results[j, i] = "cross"
                            results[i, j] = "cross"
                            results_cross[i, j] = percentile
                            results_cross[j, i] = percentile
                        else:
                            results[j, i] = "cross"
                            results[i, j] = "cross"
                            results_cross[i, j] = None
                            results_cross[j, i] = None
                    else:
                        results_cross[i, j] = None
                        results_cross[j, i] = None
        
        # Convert to DataFrames
        results_df = pd.DataFrame(results, columns=column_names, index=column_names)
        np.fill_diagonal(results_df.values, "")
        print("Critical ratios")
        display(results_df)
        results_cross_df = pd.DataFrame(results_cross, columns=column_names, index=column_names)
        np.fill_diagonal(results_cross_df.values, "")

        results_cross_display = results_cross_df.fillna("No")
        print("Crossing points before the poverty line")
        display(results_cross_display)


        if gamma:
            results_df_gamma = pd.DataFrame(results_gamma, columns=column_names, index=column_names)
            print("Critical Percentiles at gamma level")
            display(results_df_gamma)
        
        
    def test_dominance_all(self):
        '''
        Test fractional dominance order for each pair of goods in X.
        Also returns crossing locations (percentile of first crossing).
        '''
        column_names = self.df.columns
        n = self.X.shape[1]
        results = np.full((n, n), "", dtype=object)
        crossing_points = np.full((n, n), np.nan, dtype=float)

        for j in range(n):
            for i in range(j):
                if self.fractional_param:
                    curve_j = self.CC[j, :]
                    curve_i = self.CC[i, :]
                else:
                    curve_j = self.C[j, :]
                    curve_i = self.C[i, :]
                diff = curve_j - curve_i
                diff_slice = diff[100:-100]

                # dominance test
                if np.all(diff_slice >= 0):
                    results[j, i] = 1
                    results[i, j] = 0
                elif np.all(diff_slice <= 0):
                    results[j, i] = 0
                    results[i, j] = 1
                else:
                    results[j, i] = "cross"
                    results[i, j] = "cross"

                    # detect crossing point (the 1st)
                    signs = np.sign(diff_slice)
                    sign_changes = np.where(np.diff(signs) != 0)[0]
                    if sign_changes.size > 0:
                        first_cross = sign_changes[0] + 100  # adjust for slicing
                        percentile = first_cross / len(self.y)
                        crossing_points[i, j] = percentile
                        crossing_points[j, i] = percentile

        dominance_df = pd.DataFrame(results, columns=column_names, index=column_names)
        crossing_df = pd.DataFrame(
            crossing_points,
            columns=column_names,
            index=column_names
        )

        print("Dominance results")
        display(dominance_df)
        print("Crossing locations (percentile of first crossing)")
        display(crossing_df.fillna(" "))

        return dominance_df, crossing_df    
