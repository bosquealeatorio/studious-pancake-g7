from utils import (
    load_data, preprocess_input, 
    find_best_model, load_params
)

if __name__ == "__main__":
    
    params=load_params()

    data=load_data(target_column=params['target_column'])
    
    X_train, X_test, y_train, y_test = preprocess_input(
        data=data, 
        target_column=params['target_column'],
        params_preprocessing=params['preprocessing']
    )

    model=find_best_model(
        X_train=X_train, 
        y_train=y_train,
        params_modeling=params['modeling']
    )
