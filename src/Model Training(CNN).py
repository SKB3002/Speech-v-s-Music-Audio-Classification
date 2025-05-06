from tensorflow.keras.optimizers import Adam

model.compile(optimizer=Adam(learning_rate = 0.0009), loss='binary_crossentropy', metrics=['accuracy'])

from sklearn.utils.class_weight import compute_class_weight

class_weights = compute_class_weight('balanced', classes=np.unique(y_train), y=y_train)
class_weight_dict = dict(enumerate(class_weights))

history = model.fit(
    X_train, y_train,
    epochs=50,
    batch_size=16,
    callbacks = [early_stop, reduce_lr],
    class_weight=class_weight_dict,
    validation_data=(X_test, y_test)
)
