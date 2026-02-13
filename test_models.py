import google.generativeai as genai

genai.configure(api_key="AIzaSyB1eT6IlQlhx6e_qAvTyJkxuTVxFP0WJAg")

for m in genai.list_models():
    print(m.name, "->", m.supported_generation_methods)
