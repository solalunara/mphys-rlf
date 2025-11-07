import sys
import os

# Add the src directory to Python path so we can import modules
src_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, src_dir)

import model.sampler
import plotting.image_plots

class SimpleSampler:
    def __init__(self):
        # Hook into the in-built sampler
        self.model_sampler = model.sampler.Sampler()

    def run(self, model_name, file_name=None, distribute_model=False, **kwargs):
        """
        Generate a single sample according to some parameters

        Parameters
        ----------
        model_name : str
            The name of the model to use for sampling.
                Should be stored in model_results/<model_name>/parameters_<model_name>.pt,
                with a config in model_results/<model_name>/config_<model_name>.json
                example configs in src/model/configs
        file_name : str | None
            Relative or global file to save the sampled image grid(s) to
        distribute_model : Bool
            If true, try to run the model on as many GPUs are avaliable (uses DistributedDataParallel)
            If false, run on the CPU (for systems without cuda or gpu support)


        kwargs
        ------

        model : torch.nn.Module, optional
            The model to use for sampling. If not provided, the model will be loaded using `model_utils.load_model`.
        context : torch.Tensor, optional
            The context tensor for conditioning the sampling. If provided, it should have shape (n_samples, context_dim).
        context_fn : callable, optional
            A function that generates the context tensor for conditioning the sampling. If provided, it should take the number of samples as input and return a tensor of shape (n_samples, context_dim).
        labels : torch.Tensor, optional
            The labels tensor for conditioning the sampling. If provided, it should have shape (n_samples, label_dim).
        latents : torch.Tensor, optional
            The latents tensor for conditioning the sampling. If provided, it should have shape (n_samples, latent_dim).
        distribute_model : bool, optional
            Whether to distribute the model across multiple devices. Defaults to True.
        device_ids : list of int, optional
            The device IDs to use for distributing the model. If not provided, the available devices will be used.
        model_kwargs : dict, optional
            Additional keyword arguments to pass to `model_utils.load_model` when loading the model.
        **settings_kwargs : dict
            Additional keyword arguments to update the sampler settings, which are:

                Sampling setup:
                "n_samples": 1000,
                "n_devices": 1,
                "samples_per_device": 1000,  # Depending on model size
                "image_size": 80,

                Output setup:
                "comment": "",
                "return_steps": True,

                Solver setup:
                "timesteps": 25,
                "guidance_strength": 0.1,
                "sigma_min": 2e-3,
                "sigma_max": 80,
                "rho": 7,
                "S_churn": 0,
                "S_min": 0,
                "S_max": torch.inf,
                "S_noise": 1,
        """
        # Generate a sample according to some parameters
        samples = self.model_sampler.quick_sample(model_name, **kwargs, n_samples=1, distribute_model=distribute_model, image_size=80)

        # Save to a file
        fig, ax = plotting.image_plots.plot_image_grid(samples[0])
        if file_name is not None:
            fig.savefig(file_name)

        return samples

    def quick_run(self):
        """
        Generate a sample with default parameters for this project
        """
        return self.run(model_name="LOFAR_model", file_name="sample.png", distribute_model=False)


if __name__ == "__main__":
    simple_sample = SimpleSampler()
    simple_sample.quick_run()