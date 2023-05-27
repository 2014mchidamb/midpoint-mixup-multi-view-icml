import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt


def plot_mixup_error(
    task,
    model_type,
    optimizer,
    lr,
    n_features,
    num_runs,
    num_epochs,
    base_mixup_avg_errors,
    base_mixup_error_std,
    mixup_avg_errors,
    mixup_error_std,
    base_avg_errors,
    base_error_std,
    test_interval=1,
    error_type="Test",
    filename_append="default",
):
    plot_title = r"Error Curves, {} ({} Features, {}, {} Runs)".format(
        task, n_features, optimizer, num_runs
    )
    image_title = "plots_cat_channel/{}_{}_{}_lr_{}_{}_features_{}_runs_{}.png".format(
        task, model_type, optimizer, lr, n_features, num_runs, filename_append
    )

    # Create error plot.
    plt.figure(figsize=(9, 7))
    plt.rc("axes", titlesize=18, labelsize=18)
    plt.rc("xtick", labelsize=15)
    plt.rc("ytick", labelsize=15)
    plt.rc("legend", fontsize=18)
    plt.rc("figure", titlesize=18)

    plt.title(plot_title)
    plt.xlabel("Epochs")
    plt.ylabel("{} Error".format(error_type))

    x_epochs = [i for i in range(1, num_epochs + 1) if (i % test_interval == 0)]
    plt.plot(x_epochs, base_avg_errors, label="ERM", color="C0")
    plt.plot(x_epochs, mixup_avg_errors, label="Midpoint Mixup", color="C1")
    plt.plot(x_epochs, base_mixup_avg_errors, label="Uniform Mixup", color="C2")
    plt.fill_between(
        x_epochs,
        base_avg_errors - base_error_std,
        base_avg_errors + base_error_std,
        facecolor="C0",
        alpha=0.3,
    )
    plt.fill_between(
        x_epochs,
        mixup_avg_errors - mixup_error_std,
        mixup_avg_errors + mixup_error_std,
        facecolor="C1",
        alpha=0.3,
    )
    plt.fill_between(
        x_epochs,
        base_mixup_avg_errors - base_mixup_error_std,
        base_mixup_avg_errors + base_mixup_error_std,
        facecolor="C2",
        alpha=0.3,
    )

    plt.legend(loc="lower left")

    plt.savefig(image_title)