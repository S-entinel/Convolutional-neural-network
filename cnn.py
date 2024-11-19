import numpy as np
from typing import List, Tuple, Dict, Union
import pickle

class Layer:
    def forward(self, input_data: np.ndarray) -> np.ndarray:
        raise NotImplementedError
        
    def backward(self, grad_output: np.ndarray) -> np.ndarray:
        raise NotImplementedError
        
    def get_params_and_gradients(self) -> List[Tuple[np.ndarray, np.ndarray]]:
        return []

class ConvLayer(Layer):
    def __init__(self, n_filters: int, filter_size: int, stride: int = 1, padding: int = 0):
        self.n_filters = n_filters
        self.filter_size = filter_size
        self.stride = stride
        self.padding = padding
        self.W = None
        self.b = None
        self.W_grad = None
        self.b_grad = None
        self.input_data = None
        
    def initialize(self, input_channels: int):
        # He initialization
        self.W = np.random.randn(
            self.n_filters, input_channels, self.filter_size, self.filter_size
        ) * np.sqrt(2.0 / (input_channels * self.filter_size * self.filter_size))
        self.b = np.zeros((self.n_filters, 1))
    
    def _pad_input(self, input_data: np.ndarray) -> np.ndarray:
        if self.padding == 0:
            return input_data
        
        pad_width = ((0, 0), (0, 0), (self.padding, self.padding), (self.padding, self.padding))
        return np.pad(input_data, pad_width, mode='constant', constant_values=0)
    
    def _get_output_shape(self, input_shape: Tuple) -> Tuple:
        n, c, h, w = input_shape
        h_out = (h + 2 * self.padding - self.filter_size) // self.stride + 1
        w_out = (w + 2 * self.padding - self.filter_size) // self.stride + 1
        return (n, self.n_filters, h_out, w_out)
    
    def forward(self, input_data: np.ndarray) -> np.ndarray:
        self.input_data = input_data
        n_samples, channels, height, width = input_data.shape
        
        # Pad input if necessary
        padded_input = self._pad_input(input_data)
        
        # Calculate output dimensions
        h_out = (height + 2 * self.padding - self.filter_size) // self.stride + 1
        w_out = (width + 2 * self.padding - self.filter_size) // self.stride + 1
        
        # Initialize output
        output = np.zeros((n_samples, self.n_filters, h_out, w_out))
        
        # Perform convolution
        for i in range(h_out):
            for j in range(w_out):
                h_start = i * self.stride
                h_end = h_start + self.filter_size
                w_start = j * self.stride
                w_end = w_start + self.filter_size
                
                receptive_field = padded_input[:, :, h_start:h_end, w_start:w_end]
                
                for k in range(self.n_filters):
                    output[:, k, i, j] = np.sum(
                        receptive_field * self.W[k, :, :, :], axis=(1, 2, 3)
                    ) + self.b[k]
        
        return output
    
    def backward(self, grad_output: np.ndarray) -> np.ndarray:
        n_samples = self.input_data.shape[0]
        
        # Initialize gradients
        self.W_grad = np.zeros_like(self.W)
        self.b_grad = np.zeros_like(self.b)
        grad_input = np.zeros_like(self.input_data)
        
        # Pad input and gradient for convolution
        padded_input = self._pad_input(self.input_data)
        
        # Calculate gradients
        for i in range(grad_output.shape[2]):
            for j in range(grad_output.shape[3]):
                h_start = i * self.stride
                h_end = h_start + self.filter_size
                w_start = j * self.stride
                w_end = w_start + self.filter_size
                
                receptive_field = padded_input[:, :, h_start:h_end, w_start:w_end]
                
                for k in range(self.n_filters):
                    self.W_grad[k] += np.sum(
                        receptive_field * grad_output[:, k, i, j][:, None, None, None],
                        axis=0
                    )
                    self.b_grad[k] += np.sum(grad_output[:, k, i, j])
                    
                    grad_input[:, :, h_start:h_end, w_start:w_end] += \
                        self.W[k] * grad_output[:, k, i, j][:, None, None, None]
        
        # Remove padding from gradient if necessary
        if self.padding > 0:
            grad_input = grad_input[:, :, self.padding:-self.padding, self.padding:-self.padding]
        
        return grad_input
    
    def get_params_and_gradients(self) -> List[Tuple[np.ndarray, np.ndarray]]:
        return [(self.W, self.W_grad), (self.b, self.b_grad)]

class MaxPoolLayer(Layer):
    def __init__(self, pool_size: int, stride: int = None):
        self.pool_size = pool_size
        self.stride = stride if stride is not None else pool_size
        self.input_data = None
        self.mask = None
    
    def forward(self, input_data: np.ndarray) -> np.ndarray:
        self.input_data = input_data
        n_samples, channels, height, width = input_data.shape
        
        h_out = (height - self.pool_size) // self.stride + 1
        w_out = (width - self.pool_size) // self.stride + 1
        
        output = np.zeros((n_samples, channels, h_out, w_out))
        self.mask = np.zeros_like(input_data)
        
        for i in range(h_out):
            for j in range(w_out):
                h_start = i * self.stride
                h_end = h_start + self.pool_size
                w_start = j * self.stride
                w_end = w_start + self.pool_size
                
                receptive_field = input_data[:, :, h_start:h_end, w_start:w_end]
                output[:, :, i, j] = np.max(receptive_field, axis=(2, 3))
                
                # Store pooling indices for backprop
                max_indices = np.argmax(receptive_field.reshape(n_samples, channels, -1), axis=2)
                for sample in range(n_samples):
                    for channel in range(channels):
                        flat_idx = max_indices[sample, channel]
                        h_idx = flat_idx // self.pool_size
                        w_idx = flat_idx % self.pool_size
                        self.mask[sample, channel, h_start+h_idx, w_start+w_idx] = 1
        
        return output
    
    def backward(self, grad_output: np.ndarray) -> np.ndarray:
        grad_input = np.zeros_like(self.input_data)
        
        _, _, h_out, w_out = grad_output.shape
        
        for i in range(h_out):
            for j in range(w_out):
                h_start = i * self.stride
                h_end = h_start + self.pool_size
                w_start = j * self.stride
                w_end = w_start + self.pool_size
                
                grad_input[:, :, h_start:h_end, w_start:w_end] += \
                    self.mask[:, :, h_start:h_end, w_start:w_end] * \
                    grad_output[:, :, i, j][:, :, None, None]
        
        return grad_input

class ReLU(Layer):
    def __init__(self):
        self.input_data = None
    
    def forward(self, input_data: np.ndarray) -> np.ndarray:
        self.input_data = input_data
        return np.maximum(0, input_data)
    
    def backward(self, grad_output: np.ndarray) -> np.ndarray:
        return grad_output * (self.input_data > 0)

class Flatten(Layer):
    def __init__(self):
        self.input_shape = None
    
    def forward(self, input_data: np.ndarray) -> np.ndarray:
        self.input_shape = input_data.shape
        return input_data.reshape(input_data.shape[0], -1)
    
    def backward(self, grad_output: np.ndarray) -> np.ndarray:
        return grad_output.reshape(self.input_shape)

class FullyConnected(Layer):
    def __init__(self, output_size: int):
        self.output_size = output_size
        self.W = None
        self.b = None
        self.W_grad = None
        self.b_grad = None
        self.input_data = None
    
    def initialize(self, input_size: int):
        # He initialization
        self.W = np.random.randn(self.output_size, input_size) * np.sqrt(2.0 / input_size)
        self.b = np.zeros((self.output_size, 1))
    
    def forward(self, input_data: np.ndarray) -> np.ndarray:
        self.input_data = input_data
        return np.dot(input_data, self.W.T) + self.b.T
    
    def backward(self, grad_output: np.ndarray) -> np.ndarray:
        self.W_grad = np.dot(grad_output.T, self.input_data)
        self.b_grad = np.sum(grad_output, axis=0, keepdims=True).T
        return np.dot(grad_output, self.W)
    
    def get_params_and_gradients(self) -> List[Tuple[np.ndarray, np.ndarray]]:
        return [(self.W, self.W_grad), (self.b, self.b_grad)]

class Softmax:
    def forward(self, input_data: np.ndarray) -> np.ndarray:
        exp_data = np.exp(input_data - np.max(input_data, axis=1, keepdims=True))
        return exp_data / np.sum(exp_data, axis=1, keepdims=True)

class CrossEntropyLoss:
    def forward(self, predicted: np.ndarray, actual: np.ndarray) -> float:
        m = predicted.shape[0]
        log_likelihood = -np.log(predicted[range(m), actual])
        return np.sum(log_likelihood) / m
    
    def backward(self, predicted: np.ndarray, actual: np.ndarray) -> np.ndarray:
        m = predicted.shape[0]
        grad = predicted.copy()
        grad[range(m), actual] -= 1
        return grad / m

class CNN:
    def __init__(self):
        self.layers: List[Layer] = []
        self.softmax = Softmax()
        self.loss_function = CrossEntropyLoss()
    
    def add_layer(self, layer: Layer):
        self.layers.append(layer)
    
    def _initialize_layers(self, input_shape: Tuple):
        current_shape = input_shape
        
        for layer in self.layers:
            if isinstance(layer, ConvLayer):
                layer.initialize(current_shape[1])
                current_shape = layer._get_output_shape(current_shape)
            elif isinstance(layer, MaxPoolLayer):
                h_out = (current_shape[2] - layer.pool_size) // layer.stride + 1
                w_out = (current_shape[3] - layer.pool_size) // layer.stride + 1
                current_shape = (current_shape[0], current_shape[1], h_out, w_out)
            elif isinstance(layer, Flatten):
                current_shape = (current_shape[0], np.prod(current_shape[1:]))
            elif isinstance(layer, FullyConnected):
                layer.initialize(current_shape[1])
                current_shape = (current_shape[0], layer.output_size)
    
    def forward(self, X: np.ndarray, training: bool = True) -> np.ndarray:
        output = X
        for layer in self.layers:
            output = layer.forward(output)
        return self.softmax.forward(output)
    
    def backward(self, grad: np.ndarray) -> None:
        for layer in reversed(self.layers):
            grad = layer.backward(grad)
    
    def train(self, 
             X_train: np.ndarray, 
             y_train: np.ndarray, 
             X_val: np.ndarray = None,
             y_val: np.ndarray = None,
             epochs: int = 10, 
             batch_size: int = 32, 
             learning_rate: float = 0.01,
             learning_rate_decay: float = 0.95,
             momentum: float = 0.9) -> Dict[str, List[float]]:
        
        if len(X_train.shape) != 4:
            raise ValueError("Input data must have shape (n_samples, channels, height, width)")
        
        # Initialize layers
        self._initialize_layers(X_train.shape)
        
        n_samples = X_train.shape[0]
        history = {
            'train_loss': [],
            'train_acc': [],
            'val_loss': [],
            'val_acc': []
        }
        
        # Initialize velocity for momentum
        velocity = []
        for layer in self.layers:
            layer_velocity = []
            for param, _ in layer.get_params_and_gradients():
                layer_velocity.append(np.zeros_like(param))
            velocity.append(layer_velocity)
        
        for epoch in range(epochs):
            # Shuffle training data
            indices = np.random.permutation(n_samples)
            X_train = X_train[indices]
            y_train = y_train[indices]
            
            total_loss = 0
            correct_predictions = 0
            
            # Mini-batch training
            for i in range(0, n_samples, batch_size):
                batch_X = X_train[i:i+batch_size]
                batch_y = y_train[i:i+batch_size]
                
                # Forward pass
                predictions = self.forward(batch_X)
                
                # Calculate loss
                loss = self.loss_function.forward(predictions, batch_y)
                total_loss += loss * len(batch_y)
                
                # Calculate accuracy
                predicted_classes = np.argmax(predictions, axis=1)
                correct_predictions += np.sum(predicted_classes == batch_y)
                
                # Backward pass
                grad = self.loss_function.backward(predictions, batch_y)
                self.backward(grad)
                
                # Update parameters with momentum
                for layer_idx, layer in enumerate(self.layers):
                    velocity_idx = 0
                    for param, grad in layer.get_params_and_gradients():
                        if param is not None:
                            velocity[layer_idx][velocity_idx] = (
                                momentum * velocity[layer_idx][velocity_idx] - 
                                learning_rate * grad
                            )
                            param += velocity[layer_idx][velocity_idx]
                            velocity_idx += 1
            
            # Calculate training metrics
            train_loss = total_loss / n_samples
            train_accuracy = correct_predictions / n_samples
            history['train_loss'].append(train_loss)
            history['train_acc'].append(train_accuracy)
            
            # Validation step
            if X_val is not None and y_val is not None:
                val_predictions = self.forward(X_val, training=False)
                val_loss = self.loss_function.forward(val_predictions, y_val)
                val_predicted_classes = np.argmax(val_predictions, axis=1)
                val_accuracy = np.mean(val_predicted_classes == y_val)
                
                history['val_loss'].append(val_loss)
                history['val_acc'].append(val_accuracy)
                
                print(f"Epoch {epoch + 1}/{epochs}")
                print(f"Train Loss: {train_loss:.4f} - Train Acc: {train_accuracy:.4f}")
                print(f"Val Loss: {val_loss:.4f} - Val Acc: {val_accuracy:.4f}")
            else:
                print(f"Epoch {epoch + 1}/{epochs}")
                print(f"Train Loss: {train_loss:.4f} - Train Acc: {train_accuracy:.4f}")
            
            # Decay learning rate
            learning_rate *= learning_rate_decay
        
        return history
    
    def evaluate(self, X: np.ndarray, y: np.ndarray) -> Tuple[float, float]:
        """
        Evaluate the model on test data
        """
        predictions = self.forward(X, training=False)
        loss = self.loss_function.forward(predictions, y)
        predicted_classes = np.argmax(predictions, axis=1)
        accuracy = np.mean(predicted_classes == y)
        return loss, accuracy
    
    def predict(self, X: np.ndarray) -> np.ndarray:
        """
        Make predictions on new data
        """
        predictions = self.forward(X, training=False)
        return np.argmax(predictions, axis=1)
    
    def save_model(self, filename: str):
        """
        Save model parameters to file
        """
        model_params = []
        for layer in self.layers:
            layer_params = []
            for param, _ in layer.get_params_and_gradients():
                if param is not None:
                    layer_params.append(param)
            model_params.append(layer_params)
        
        with open(filename, 'wb') as f:
            pickle.dump(model_params, f)
    
    def load_model(self, filename: str):
        """
        Load model parameters from file
        """
        with open(filename, 'rb') as f:
            model_params = pickle.load(f)
        
        for layer_idx, layer in enumerate(self.layers):
            param_idx = 0
            for param, _ in layer.get_params_and_gradients():
                if param is not None:
                    param[:] = model_params[layer_idx][param_idx]
                    param_idx += 1

def preprocess_data(X: np.ndarray, mean: np.ndarray = None, std: np.ndarray = None) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Preprocess input data with zero mean and unit variance
    """
    if mean is None:
        mean = np.mean(X, axis=0)
    if std is None:
        std = np.std(X, axis=0)
    
    X_normalized = (X - mean) / (std + 1e-8)
    return X_normalized, mean, std

# Example usage:
def create_simple_cnn(input_shape: Tuple[int, int, int], num_classes: int) -> CNN:
    """
    Create a simple CNN architecture
    """
    cnn = CNN()
    
    # First convolutional block
    cnn.add_layer(ConvLayer(n_filters=32, filter_size=3, padding=1))
    cnn.add_layer(ReLU())
    cnn.add_layer(MaxPoolLayer(pool_size=2))
    
    # Second convolutional block
    cnn.add_layer(ConvLayer(n_filters=64, filter_size=3, padding=1))
    cnn.add_layer(ReLU())
    cnn.add_layer(MaxPoolLayer(pool_size=2))
    
    # Fully connected layers
    cnn.add_layer(Flatten())
    cnn.add_layer(FullyConnected(output_size=128))
    cnn.add_layer(ReLU())
    cnn.add_layer(FullyConnected(output_size=num_classes))
    
    return cnn

# Usage example:
if __name__ == "__main__":
    # Generate dummy data for demonstration
    np.random.seed(42)
    X_train = np.random.randn(1000, 3, 32, 32)  # 1000 RGB images of size 32x32
    y_train = np.random.randint(0, 10, size=1000)  # 10 classes
    X_val = np.random.randn(200, 3, 32, 32)
    y_val = np.random.randint(0, 10, size=200)
    
    # Preprocess data
    X_train, mean, std = preprocess_data(X_train)
    X_val, _, _ = preprocess_data(X_val, mean, std)
    
    # Create and train model
    cnn = create_simple_cnn(input_shape=(3, 32, 32), num_classes=10)
    history = cnn.train(
        X_train=X_train,
        y_train=y_train,
        X_val=X_val,
        y_val=y_val,
        epochs=10,
        batch_size=32,
        learning_rate=0.01,
        learning_rate_decay=0.95,
        momentum=0.9
    )
    
    # Save model
    cnn.save_model('cnn_model.pkl')