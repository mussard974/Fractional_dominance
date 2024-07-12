import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.cm as cm

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

    def fit(self, X, y, dominance_param = True, fractional_param = None):
        
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
        X_all = np.hstack((y, X))
        X_all = X_all[X_all[:,0].argsort()]  
        y = X_all[:,0]
        self.y = y
        n,_ = X.shape
        self.n = n
        x = X_all[:,1::]                # consumption
        x = x.astype(float)
        self.x = x
        a = x / x.mean(axis=0, keepdims=True)
        self.a = a 
        t1 = np.ones((n,1))*(1/n)
        t1 = np.cumsum(t1)
        self.t1 = t1

        C = np.zeros((x.shape[1], n, self.dominance_param))   # CD-CURVES of ORDER 2,3,...
        for col in range(x.shape[1]):
            C[col,0,0] = t1[0]*a[0,col]/2
            for i in range (1,self.dominance_param):           
                C[col,0,i] = t1[0]*C[col,0,i-1]/2
        for col in range(x.shape[1]):
            for i in range(1,n):
                C[col,i,0] = C[col,i-1,0] + ((a[i-1,col] + a[i,col])*(t1[i]-t1[i-1]))/2  # Order 2 
                for j in range(1,self.dominance_param):
                    C[col,i,j] = C[col,i-1,j] + ((C[col,i-1,j-1] + C[col,i,j-1])*(t1[i]-t1[i-1]))/2 # order 3
        self.C = C[:,:,self.dominance_param-2]
        self.C_all_order = C
        
        if fractional_param is not None:
            
            CC = np.zeros((x.shape[1],n,self.dominance_param)) # CD-CURVES with interpolations
            for col in range(x.shape[1]):
                CC[col,0,0] = a[0,col]*np.exp((1/self.fractional_param-1)*t1[0])*t1[0]/2
            for col in range(x.shape[1]):
                for j in range(1,self.dominance_param):          
                    CC[col,0,j] = C[col,0,j-1]*np.exp((1/self.fractional_param-1)*t1[0])*t1[0]/2
            for col in range(x.shape[1]):
                for i in range(1,n):
                    CC[col,i,0] = CC[col,i-1,0] + ((np.exp((1/self.fractional_param-1)*t1[i])*t1[i] - np.exp((1/self.fractional_param-1)*t1[i-1])*t1[i-1])*(a[i,col]+a[i-1,col]))/2  # order 2 
                    for j in range(1,self.dominance_param): 
                        CC[col,i,j] = CC[col,i-1,j] + ((np.exp((1/self.fractional_param-1)*t1[i])*t1[i] - np.exp((1/self.fractional_param-1)*t1[i-1])*t1[i-1])*(C[col,i,j-1]+C[col,i-1,j-1]))/2 # order 3
            self.CC = CC[:,:,self.dominance_param-1]
        
        if fractional_param is not None:
            return CC[:,:,self.dominance_param-1]
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
        for fractional_param in np.arange(0.01, 1, 0.01):
            CC = np.zeros((k, n))
            if self.dominance_param == 1:
                for col in range(k):
                    CC[col, 0] = self.a[0, col] * np.exp((1 / fractional_param - 1) * self.t1[0]) * self.t1[0] / 2
                    for i in range(1, n):
                        CC[col, i] = CC[col, i-1] + ((np.exp((1 / fractional_param - 1) * self.t1[i]) * self.t1[i] - 
                                                            np.exp((1 / fractional_param - 1) * self.t1[i-1]) * self.t1[i-1]) * 
                                                        (self.a[i, col] + self.a[i-1, col])) / 2
            else:
                for col in range(k):
                    CC[col, 0] = self.C_all_order[col, 0, self.dominance_param-1] * np.exp((1 / fractional_param - 1) * self.t1[0]) * self.t1[0] / 2
                    for i in range(1, n):
                        CC[col, i] = CC[col, i-1] + ((np.exp((1 / fractional_param - 1) * self.t1[i]) * self.t1[i] - 
                                                            np.exp((1 / fractional_param - 1) * self.t1[i-1]) * self.t1[i-1]) * 
                                                        (self.C_all_order[col, i, self.dominance_param-1] + self.C_all_order[col, i-1, self.dominance_param-1])) / 2
            
            for j in range (self.X.shape[1]):
                for i in range(j):
                    diff = CC[j,:] - CC[i,:]
                    if frac == 0:
                        if all(x >= 0 for x in diff[100:-100]):
                            results_bis[j,i] = self.dominance_param + fractional_param
                        if all(x <= 0 for x in diff[100:-100]):
                            results_bis[j,i] = self.dominance_param + fractional_param
                    else:    
                        if all(x >= 0 for x in diff[100:-100]) and results_bis[j,i] == 0.0:
                                results_bis[j,i] = self.dominance_param + fractional_param
                        if all(x <= 0 for x in diff[100:-100]) and results_bis[j,i] == 0.0:
                                results_bis[j,i] = self.dominance_param + fractional_param
            frac = frac + 1
        #results_bis = np.where(results_bis == None, "", results_bis)
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
            n = self.X.shape[0]
            t1 = np.ones((n,1))*(1/n)
            t1 = np.cumsum(t1)
            plt.plot(t1, self.CC[position_good_1,:], 'b', label=good_1)
            plt.plot(t1, self.CC[position_good_2,:], 'r', label=good_2)
            plt.ylabel('Cumulative consumption')
            plt.xlabel('Income percentiles')
            plt.title(f's-Curves of order {self.dominance_param}+{self.fractional_param}')
            plt.legend(loc='best')
            plt.show()            
        else:    
            n = self.X.shape[0]
            t1 = np.ones((n,1))*(1/n)
            t1 = np.cumsum(t1)
            plt.plot(t1, self.C[position_good_1,:], 'b', label=good_1)
            plt.plot(t1, self.C[position_good_2,:], 'r', label=good_2)
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
            n = self.X.shape[0]
            t1 = np.ones((n,1))*(1/n)
            t1 = np.cumsum(t1)
            colormap = cm.get_cmap('rainbow', self.CC.shape[0])
            for i in range(self.CC.shape[0]):
                plt.plot(t1, self.CC[i, :], color=colormap(i), label=self.df.columns[i])
            plt.ylabel('Cumulative consumption')
            plt.xlabel('Income percentiles')
            plt.title(f's-Curves of order {self.dominance_param}+{self.fractional_param}')
            plt.legend(loc='best')
            plt.show()            
        else:    
            n = self.X.shape[0]
            t1 = np.ones((n,1))*(1/n)
            t1 = np.cumsum(t1)
            colormap = cm.get_cmap('rainbow', self.C.shape[0])
            for i in range(self.C.shape[0]):
                plt.plot(t1, self.C[i, :], color=colormap(i), label=self.df.columns[i])
            plt.ylabel('Cumulative consumption')
            plt.xlabel('Income percentiles')
            plt.title(f's-Curves of order {self.dominance_param}')
            plt.legend(loc='best')
            plt.savefig('high_resolution_plot.png', dpi=500)
            plt.show()
            
    def bootstrap(self, curve, confidence_interval = 0.9, B = 5000):    
        n = self.n 
        IC = np.zeros((n,2))
        Boot = np.zeros((n,B))
        for i in range(B):
            Boot[:,i] = np.sort(np.random.choice(curve, size=n, replace=True), axis=0)
        for i in range(n):
            Boot[i,:] = np.sort(Boot[i,:])
            IC[i,0], IC[i,1] = Boot[i,int(B*(1-confidence_interval)/2)], Boot[i,int(B*(confidence_interval/2))]    
        return IC[:,0], IC[:,1]
    
    def efficiency_gain(self, good_1, good_2):
        position_good_1 = self.df.columns.get_loc(good_1)
        position_good_2 = self.df.columns.get_loc(good_2)
        if self.fractional_param:
            curve1 = self.CC[position_good_1,:]
            curve2 = self.CC[position_good_2,:]
        else:
            curve1 = self.C[position_good_1,:]
            curve2 = self.C[position_good_2,:]
        
        n = self.X.shape[0]
        t1 = np.ones((n,1))*(1/n)
        t1 = np.cumsum(t1)
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
            x1, x2 = t1[index], t1[index + 1]
            y1, y2 = differences[index], differences[index + 1]
            intersection_x = x1 - y1 * (x2 - x1) / (y2 - y1)
            intersection_points.append(intersection_x)
            
            # Find the income value corresponding to the intersection point
            income_value = np.interp(intersection_x, t1, self.y)
            income_values_at_intersections.append(income_value)

        if not intersection_points:
            print("No intersection points found")
        else:
            for i, point in enumerate(intersection_points):
                print(f'Intersection point {i+1}: p={point}, Income={income_values_at_intersections[i]}')
        
        print("Efficiency gain=", np.amin(delta2))
        print("Efficiency gain=", delta2)

        # Plot the curves
        plt.plot(t1, curve1, 'r', label=good_1)
        plt.plot(t1, curve2, 'g', label=good_2)
        plt.plot(t1, delta, 'b', label='Benefits')
        plt.ylabel('Cumulative distribution')
        plt.xlabel('Incomes')
        plt.title('CD-Curves')
        plt.legend(loc='best')
        plt.show()
        
    def critical_ratios_poverty(self, gamma = None):
        '''
        Compute CD_i / CD_j to find the minimal value of gamma_ij at the poverty line
        '''
        np.seterr(divide='ignore', invalid='ignore')
        column_names = self.df.columns
        results = np.full((self.X.shape[1], self.X.shape[1]), "", dtype=object)
        results_gamma = np.full((self.X.shape[1], self.X.shape[1]), "", dtype=object)
        results_cross = np.full((self.X.shape[1], self.X.shape[1]), "", dtype=object)
        
        # Poverty line
        z = np.median(self.y)*0.5
        sorted_y = np.sort(self.y)
        self.rank_poverty = np.sum(sorted_y < z)
        self.p_z = np.sum(sorted_y < z) / len(self.y)
        
        for j in range(self.X.shape[1]):
            for i in range(j):
                if self.fractional_param:
                    curve_j = self.CC[j, :]
                    curve_i = self.CC[i, :]
                else:
                    curve_j = self.C[j, :]
                    curve_i = self.C[i, :]

                sliced_curve_j = curve_j[100:-100]
                sliced_curve_i = curve_i[100:-100]
                diff = sliced_curve_j - sliced_curve_i
                ratio = sliced_curve_j / sliced_curve_i
                delta = ratio[self.rank_poverty]
                                
                if all(diff >= 0):
                    results[j, i] = delta
                    results[i, j] = delta
                    if gamma:
                        diffs = ratio - gamma
                        nonnegative_indices = np.where(diffs >= 0)[0]
                        if nonnegative_indices.size == 0:
                            pass
                        else:  
                            nonnegative_diffs = diffs[nonnegative_indices]
                            closest_index = np.argmin(nonnegative_diffs)
                            actual_index = nonnegative_indices[closest_index]                            
                            results_gamma[i, j] = (actual_index + 101) / len(self.y)
                            results_gamma[j, i] = (actual_index + 101) / len(self.y)

                elif all(diff <= 0):
                    if delta != 0:
                        results[j, i] = 1 / delta
                        results[i, j] = 1 / delta
                    if gamma:
                        diffs = 1/ratio - gamma
                        nonnegative_indices = np.where(diffs >= 0)[0]
                        if nonnegative_indices.size == 0:
                            pass
                        else:  
                            nonnegative_diffs = diffs[nonnegative_indices]
                            closest_index = np.argmin(nonnegative_diffs)
                            actual_index = nonnegative_indices[closest_index]                            
                            results_gamma[i, j] = (actual_index + 101) / len(self.y)
                            results_gamma[j, i] = (actual_index + 101) / len(self.y)
                else:
                    sign_changes = np.where(np.diff(np.sign(diff)))[0] # return the lowest index of the sign change of diff
                    percentile = (sign_changes[0] + 101) / len(self.y) # percentile of the little crossing point
                    results[j, i] = "cross"
                    results[i, j] = "cross"
                    results_cross[i, j] = percentile
                    results_cross[j, i] = percentile
                
        results_df = pd.DataFrame(results, columns=column_names, index=column_names)
        print("Critical ratios")
        display(results_df)
        results_cross_df = pd.DataFrame(results_cross, columns=column_names, index=column_names)
        print("Crossing points")
        display(results_cross_df)
        print("Poverty line percentile", self.p_z)
        if gamma:
            results_df_gamma = pd.DataFrame(results_gamma, columns=column_names, index=column_names)
            print("Critical Percentiles at gamma level")
            display(results_df_gamma)
        
    def test_dominance_all(self):
        '''
        test fractional dominance order for each and every pairs of goods in X
         
        '''
        column_names = self.df.columns
        results = np.full((self.X.shape[1], self.X.shape[1]), None, dtype=object)
        for j in range (self.X.shape[1]):
            for i in range(j):
                if self.fractional_param:
                    curve_j = self.CC[j,:]
                    curve_i = self.CC[i,:]
                else:
                    curve_j = self.C[j,:]
                    curve_i = self.C[i,:]
                diff = curve_j - curve_i
                #sign_changes = np.where(np.diff(np.sign(diff)))[0]
                #if len(sign_changes) == 0:
                if all(x >= 0 for x in diff[100:-100]):
                    results[j,i] = 1
                    results[i,j] = 0
                elif all(x <= 0 for x in diff[100:-100]):
                    results[j,i] = 0
                    results[i,j] = 1
                else:
                    results[j,i] = "cross"
                    results[i,j] = "cross"
        results = np.where(results == None, "", results)
        results_df = pd.DataFrame(results, columns=column_names, index=column_names)
        return display(results_df)
    
