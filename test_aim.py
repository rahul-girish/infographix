
from aim import Run

run = Run(experiment="aim_setup_test")

run["hparams"] = {
    "learning_rate": 0.001,
    "batch_size": 16,
    "model": "test_model",
}

for epoch in range(10):
    train_loss = 1.0 / (epoch + 1)
    val_loss = 1.2 / (epoch + 1)

    run.track(
        train_loss,
        name="loss",
        step=epoch,
        context={"subset": "train"},
    )

    run.track(
        val_loss,
        name="loss",
        step=epoch,
        context={"subset": "validation"},
    )

run["status"] = "completed"
run.close()

print("Aim tracking test completed.")

