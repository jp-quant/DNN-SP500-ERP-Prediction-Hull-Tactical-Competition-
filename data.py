import pandas as pd
import numpy as np


class DataManager(object):

    def __init__(self,
                target = 'ASPFWR5',
                full_data_fname='dataset.csv',):

        self.original_df = pd.read_csv(full_data_fname,index_col=0,header=0)
        self.sub_datasets = self.initialize_subdatasets(self.original_df)

    

    def prepare_XY_multiInput(self,train_df,target,lookback):
        """
        Params:
         train_df - training dataframe for the operation, include both features and target (pd.DataFrame)
         target - target column in the dataframe given (string)
         lookback - days/hr/mins look back (int)
         batch_size - well..batch size
        
        Return: iter object that spit out batches, each batch = ([f[1],...f[bsize]], [t1,tbsize])
         F[] = [f[b_size],...,f[b_size]]
            |- f[b_size] = [input_1(lookback,n_features),...,input_bsize(lookback,n_features)]
                |- input_n(lookback,n_features) = array-form (np.array, pd.Dataframe, etc) of inputs with shape (lookback,n_features)

         T[] = [t[b_size],...,t[b_size]]
            |- t[b_size] = [output_1(n_targets,),...,output_bsize(n_targets,)]
                |- output_1(n_targets,) = array-form of outputs (prediction targets) of output with shape (n_targets,)
        """
        assert(target in list(train_df.columns))

        # THIS IS WRITTEN FOR HULL TACTICAL COMP TO PREDICT ASPFWR5
        # each input should be a dataframe of data in lookback, 
        # and output should be a (1,) array-form data of the ASPFWR5 from the last index in input

        x = []
        y = []
        start = 0
        end = lookback
        while (end != len(train_df)):
            lback_df = train_df[start:end]
            x.append(np.array(lback_df.drop([target],axis=1)))
            y.append(np.array(lback_df[[target]].iloc[-1]))
            start += 1
            end += 1
        
        return (np.array(x),np.array(y))



    def first_available_at(self,df):
        ''' PURPOSE: identify the first available data's stamp
                    for all financial variables given
            RETURN: sorted series of variables and their stamps
        '''
        variables = []
        stamps = []
        for ind in df.columns:
            variables.append(ind)
            stamps.append(df[ind].dropna().index[0])

        result = pd.Series(data = stamps, index = variables)

        return result.sort_values()


    def initialize_subdatasets(self,df,save_subdatasets=False):
        first = self.first_available_at(df)

        stamps = list(first.unique())

        columns = []
        result= []
    
        for i in range(len(stamps)-1):
            start_stamp = stamps[i]
            for s in first.index:
                if ((first[s] == start_stamp) and (s not in columns)):
                    columns.append(s)
            start_pos = list(df.index).index(start_stamp)
            selected_index = df.index[start_pos:]
            result.append(df.loc[selected_index][columns])
        
        to_return = {}

        for i in range(len(result)):
            if i == 0:
                to_return['start'] = result[i]
                if save_subdatasets is True:
                    result[i].to_csv('sub_datasets/1) start.csv')
            else:
                previous = result[i-1].columns
                current = result[i].columns
                added = '-'.join(current.drop(current&previous))
                if save_subdatasets is True:
                    f_name = str(i+1) + ") "
                    f_name = f_name + added + " added.csv"
                    result[i].to_csv('sub_datasets/' + f_name)
                to_return[added+ " added"] = result[i]
        
        return to_return
                 
