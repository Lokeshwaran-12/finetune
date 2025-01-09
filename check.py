prompt = "1. What is Data Science?"
input_ids = tokenizer(prompt, return_tensors="pt").input_ids
output_ids = model.generate(input_ids, max_length=128, eos_token_id=tokenizer.eos_token_id)
response = tokenizer.decode(output_ids[0], skip_special_tokens=True)
print(response)