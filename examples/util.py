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
    else:
        print(f"{name} - Train Loss: {train_loss}")
        print(f"{name} - Test Loss: {test_loss}\n")
    return train_loss, test_loss

def eval_multiclass_model(model, train_X, train_y, test_X, test_y,
                          name="model", output=None):
    train_acc = model.score(train_X, train_y)
    test_acc = model.score(test_X, test_y)
    if output is not None:
        with open(output, "a") as file:
            file.write(f"{name} - Train Accuracy: {train_acc}\n")
            file.write(f"{name} - Test Accuracy: {test_acc}\n\n")
    else:
        print(f"{name} - Train Accuracy: {train_acc}")
        print(f"{name} - Test Accuracy: {test_acc}\n")
    return train_acc, test_acc

def image_resize(im, pool_size=2):
    """
    Resize image using max-pooling (e.g. 28x28 image -> 14x14 image)
    """
    rows, cols = im.shape
    new_shape = (rows // 2, pool_size, cols // 2, pool_size)
    new_im = im.reshape(new_shape)
    new_im = new_im.transpose(0, 2, 1, 3)
    return new_im.max(axis=(2, 3))
