model = Sequential([
    Conv2D(32, (3,3), activation = 'relu', input_shape = (256,256,3)),
    MaxPooling2D(2,2),
    
    Conv2D(64, (3,3), activation = 'relu'),
    MaxPooling2D(2,2),

    Conv2D(128, (3,3), activation = 'relu'),
    MaxPooling2D(2,2),

    Conv2D(128, (3,3), activation = 'relu'),
    MaxPooling2D(2,2),

    Flatten(),
    Dense(512, activation = 'relu'),
    Dropout(0.5),
    Dense(train_generator.num_classes, activation = 'softmax')
])