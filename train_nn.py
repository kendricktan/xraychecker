import classifier

X, Y, X_test, Y_test = classifier.get_data_target()
model = classifier.get_model()

try:
    model.load('model.ckpt')
    print('loaded previously saved model!')
except:
    pass

model.fit(X, Y, n_epoch=1000, shuffle=True, validation_set=(X_test, Y_test),
                    show_metric=True, batch_size=64, snapshot_step=10,
                    snapshot_epoch=False, run_id='googlenet_oxflowers17')

model.save('model.ckpt')
