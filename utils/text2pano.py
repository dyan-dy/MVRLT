# check if there is pre-trained model for text2pano (e.g. skyboxAI)
# known: we have trained dataset already.
# pretrained weights: https://huggingface.co/jbilcke-hf/flux-dev-panorama-lora-2

from diffusers import StableDiffusionPipeline
import torch

def text_to_panorama(prompt, output_path="panorama_output.png", 
                      base_model_id="stabilityai/stable-diffusion-2-1-base", 
                      lora_model_id="jbilcke-hf/flux-dev-panorama-lora-2", 
                      lora_scale=1.0, num_inference_steps=30, guidance_scale=7.5):
    """
    Generate a panorama-style image using flux-dev-panorama-lora-2.

    Args:
        prompt (str): Text prompt for image generation.
        output_path (str): Path to save the generated image.
        base_model_id (str): Base Stable Diffusion model.
        lora_model_id (str): LoRA fine-tuned model on Hugging Face.
        lora_scale (float): Strength of LoRA effect.
        num_inference_steps (int): Number of denoising steps.
        guidance_scale (float): CFG scale for prompt adherence.

    Returns:
        PIL.Image: Generated panorama image.
    """
    # Load base Stable Diffusion model
    pipe = StableDiffusionPipeline.from_pretrained(
        base_model_id,
        torch_dtype=torch.float16,
    ).to("cuda")

    # Load and apply the LoRA weights
    pipe.load_lora_weights(lora_model_id)
    pipe.fuse_lora(lora_scale=lora_scale)

    # Generate the image
    result = pipe(prompt, num_inference_steps=num_inference_steps, guidance_scale=guidance_scale)
    image = result.images[0]

    # Save and return the image
    image.save(output_path)
    return image

# # Example usage:
# if __name__ == "__main__":
#     prompt = "a breathtaking 360-degree panorama of a futuristic city skyline at sunset, ultra-realistic"
#     img = generate_panorama(prompt)
#     img.show()
