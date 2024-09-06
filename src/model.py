# Import necessary modules and libraries
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense, Dropout, BatchNormalization
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import ReduceLROnPlateau, EarlyStopping
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np

# Define the emotion map
EMOTION_MAP = {
    0: ('emojis/angry.png', 'Angry'),
    1: ('emojis/disgust.png', 'Disgust'),
    2: ('emojis/fear.png', 'Fear'),
    3: ('emojis/happy.png', 'Happy'),
    4: ('emojis/sad.png', 'Sad'),
    5: ('emojis/surprise.png', 'Surprise'),
    6: ('emojis/neutral.png', 'Neutral')
}


# Function to build the CNN model
def build_model():
    model = Sequential([
        Conv2D(32, kernel_size=(3, 3), activation='relu', input_shape=(48, 48, 1)),
        BatchNormalization(),
        MaxPooling2D(pool_size=(2, 2)),
        Conv2D(64, kernel_size=(3, 3), activation='relu'),
        BatchNormalization(),
        MaxPooling2D(pool_size=(2, 2)),
        Conv2D(128, kernel_size=(3, 3), activation='relu'),
        BatchNormalization(),
        MaxPooling2D(pool_size=(2, 2)),
        Flatten(),
        Dense(512, activation='relu'),
        Dropout(0.5),
        Dense(7, activation='softmax')
    ])
    model.compile(optimizer=Adam(learning_rate=0.0001), loss='categorical_crossentropy', metrics=['accuracy'])
    return model


# Function to train the model
def train_model(model, train_generator, test_generator, epochs=50):
    reduce_lr = ReduceLROnPlateau(monitor='val_loss', factor=0.2, patience=5, min_lr=1e-7)
    early_stopping = EarlyStopping(monitor='val_loss', patience=10, restore_best_weights=True)

    history = model.fit(
        train_generator,
        epochs=epochs,
        validation_data=test_generator,
        callbacks=[reduce_lr, early_stopping]
    )
    model.save('models/emotion_recognition_model.keras')
    return history


# Function to evaluate the model
def evaluate_model(model, test_generator, X_test, y_test):
    # Predict the labels using the trained model
    y_pred = model.predict(test_generator)
    # Convert predictions from one-hot encoding to class labels
    y_pred_classes = np.argmax(y_pred, axis=1)
    y_true = np.argmax(y_test, axis=1)  # Convert true labels from one-hot encoding to class labels

    # Calculate accuracy of the model
    accuracy = accuracy_score(y_true, y_pred_classes)
    print(f'Accuracy: {accuracy:.4f}')

    # Generate a classification report
    target_names = list(EMOTION_MAP.values())
    print(classification_report(y_true, y_pred_classes, target_names=[name[1] for name in target_names]))

    # Generate and plot a confusion matrix
    cm = confusion_matrix(y_true, y_pred_classes)
    plt.figure(figsize=(10, 8))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=[name[1] for name in target_names],
                yticklabels=[name[1] for name in target_names])
    plt.xlabel('Predicted')
    plt.ylabel('True')
    plt.title('Confusion Matrix')
    plt.show()


# Main execution block
if __name__ == '__main__':
    from dataset import load_dataset, get_data_generators

    X_train, X_test, y_train, y_test = load_dataset('../data/fer2013.csv')
    train_generator, test_generator, X_test, y_test = get_data_generators(X_train, X_test, y_train, y_test)
    model = build_model()
    history = train_model(model, train_generator, test_generator, epochs=100)

    evaluate_model(model, test_generator, X_test, y_test)
