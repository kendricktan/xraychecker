import classifier

X, Y= classifier.get_data_target()
model = classifier.get_model()

model.fit(X, Y, n_epoch=1000, validation_set=0.2, shuffle=True,
                    show_metric=True, batch_size=64, snapshot_step=50,
                    snapshot_epoch=False, run_id='googlenet_oxflowers17')

model.save('model.ckpt')
