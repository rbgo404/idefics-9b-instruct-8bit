from transformers import IdeficsForVisionText2Text, AutoProcessor

class InferlessPythonModel:
	def initialize(self):
		model_id = "HuggingFaceM4/idefics-9b-instruct"
		self.model = IdeficsForVisionText2Text.from_pretrained(model_id, load_in_8bit=True, device_map="cuda")
		self.processor = AutoProcessor.from_pretrained(model_id)

	def infer(self, inputs):
		prompts = [[inputs["image_url"],inputs["prompt"]]]

		inputs = self.processor(prompts, return_tensors="pt").to("cuda")
		bad_words_ids = self.processor.tokenizer(["<image>", "<fake_token_around_image>"], add_special_tokens=False).input_ids
		generated_ids = self.model.generate(**inputs, bad_words_ids=bad_words_ids, max_length=500)
		generated_text = self.processor.batch_decode(generated_ids, skip_special_tokens=True)
		final_result = ""

		for i, t in enumerate(generated_text):
			print(f"{i}:\n{t}\n")
			final_result += f"{i}:\n{t}\n"

		return {"generated_text": final_result}

	def finalize(self):
		self.model = None
		self.processor = None