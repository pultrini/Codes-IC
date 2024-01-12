from sklearn.model_selection import train_test_split as splitter
import random

def data_separator(data, train_rate=0.70, val_rate=0.50, SEED=42):
    
    '''
        Applies train, val, test split.
    '''
    
    # Separando train, val and test IMAGE path list
    
    if not val_rate:
        train, test = splitter(data, test_size= 1-train_rate, shuffle=True, random_state=SEED)
        
        return train, test
        
    elif val_rate:
        train, test = splitter(data, test_size= 1-train_rate, shuffle=True, random_state=SEED)
        
        val, test = splitter(test, test_size= 1-val_rate, shuffle=True, random_state=SEED)

        return train, val, test 


def get_oversampling(df, num_samples):
    '''
        Returns an oversampled portion of the input dataset.
        inputs: df - df to be oversampled
                num_samples - # of samples to be added
    '''
    print(type(df))
    sampled_idxs = random.sample(range(len(df)), num_samples)
    oversampling = df.iloc[sampled_idxs]
    
    return oversampling