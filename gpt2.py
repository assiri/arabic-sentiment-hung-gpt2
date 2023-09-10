from transformers import pipeline

import time

start = time.time()
print("Time elapsed on working...")
# generator = pipeline('text-generation', model='bigscience/bloom-560m')
# generator = pipeline('text-generation', model='gpt2')
generator = pipeline("text-generation", model="EleutherAI/gpt-neo-1.3B")
# generator = pipeline('text-generation', model='EleutherAI/gpt-j-6B')
text = generator(
    "Albert Einstein was:", max_length=10, pad_token_id=50256, num_return_sequences=1
)
print(text)
time.sleep(0.9)
end = time.time()
print("Time consumed in working: ", end - start)
text = generator(
    "Albert Einstein was:", max_length=10, pad_token_id=50256, num_return_sequences=1
)
print(text)
time.sleep(0.9)
end = time.time()
print("Time consumed in working: ", end - start)
