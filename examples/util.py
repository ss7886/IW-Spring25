from sklearn.utils import shuffle as shuffle

# Shuffle and split
def split(data_X, data_y, shuffle_data=True, seed=12345, ratio=0.8):
    N, _ = data_X.shape
    if shuffle_data:
        data_X, data_y = shuffle(data_X, data_y, random_state=seed)

    split_point = int(N * ratio)
    train_X, test_X = data_X[:split_point], data_X[split_point:]
    train_y, test_y = data_y[:split_point], data_y[split_point:]
    return train_X, train_y, test_X, test_y

# Define MSE loss
def mse_loss(model, X, y):
    N, _ = X.shape
    pred = model.predict(X)
    loss = (y - pred).T @ (y - pred)
    return loss / N

# Eval model
def eval_model(model, train_X, train_y, test_X, test_y, name="model", output=None):
    train_loss = mse_loss(model, train_X, train_y)
    test_loss = mse_loss(model, test_X, test_y)
    if output is not None:
        with open(output, "a") as file:
            file.write(f"{name} - Train Loss: {train_loss}\n")
            file.write(f"{name} - Test Loss: {test_loss}\n\n")
    return train_loss, test_loss