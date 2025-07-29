from sklearn.ensemble import RandomForestRegressor

def train_model(x_train, y_train, n_estimators=100, random_state=123):
    model = RandomForestRegressor(n_estimators=n_estimators, random_state=random_state)
    model.fit(x_train, y_train)
    return model
