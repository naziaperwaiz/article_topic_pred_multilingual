
# Import libraries
import pandas as pd
from transformers import pipeline, AutoTokenizer, AutoModelForSequenceClassification
from langdetect import detect
import os

# Load the CSV file
df = pd.read_csv("ontolisst\ontolisst_samples30.csv")
print("Dataset Loaded. Sample rows:")
print(df.head())

# Define LiSST thesaurus categories
LISST_TERMS = {
    "economy": [
        "economic policies", "market stability", "financial growth",
        "trade agreements", "global economy", "investment strategies"
    ],
    "politics": [
        "elections", "government policies", "democratic systems",
        "political parties", "legislative reforms", "international relations"
    ],
    "education": [
        "education reform", "learning systems", "student welfare",
        "teacher training", "university programs", "school curriculum"
    ],
    "health": [
        "public health", "medical advancements", "mental well-being",
        "disease prevention", "healthcare access", "nutrition programs"
    ],
}

# Load a pre-trained zero-shot classification model
model_name = "facebook/bart-large-mnli"  # More suitable for zero-shot tasks
topic_classifier = pipeline("zero-shot-classification", model=model_name)

# Function to assign topics using GPT-4
def gpt4_topic_assignment(text, candidate_labels):
    try:
        prompt = f"""
        Assign the most appropriate topic from the following list to the given text:
        Topics: {', '.join(candidate_labels)}
        
        Text: {text}
        
        Respond with the topic and your confidence score in this format:
        "Topic: <topic>, Confidence: <confidence>"
        """
        response = openai.ChatCompletion.create(
            model="gpt-4",
            messages=[{"role": "user", "content": prompt}],
            max_tokens=150,  # Limit tokens to avoid exceeding the API's free quota
        )
        output = response['choices'][0]['message']['content'].strip()
        
        # Parse GPT-4's output for topic and confidence
        if "Topic:" in output and "Confidence:" in output:
            topic = output.split("Topic:")[1].split(",")[0].strip()
            confidence_str = output.split("Confidence:")[1].strip()
            
            # Clean the confidence string by removing non-numeric characters (like % or ")
            confidence_str = re.sub(r'[^0-9.]', '', confidence_str)
            
            try:
                confidence = float(confidence_str)
            except ValueError:
                confidence = 0.0
        else:
            topic = "Error"
            confidence = 0.0
        
        return topic, confidence
    except Exception as e:
        print(f"GPT-4 Error: {e}")
        return "Error", 0.0

# Function to assign topics using both NLP and GPT-4
def assign_topics(row):
    text, lang = row["text"], row["lang"]
    
    # Detect language if not provided
    detected_lang = lang if pd.notnull(lang) else detect(text)
    
    # Get candidate topics
    candidate_labels = list(LISST_TERMS.keys())
    
    # Use the NLP classifier
    nlp_result = topic_classifier(text, candidate_labels)
    nlp_topic = nlp_result["labels"][0]  # Top predicted label
    nlp_confidence = nlp_result["scores"][0]  # Confidence score
    
    # Use GPT-4
  #  gpt4_topic, gpt4_confidence = gpt4_topic_assignment(text, candidate_labels)
    
    return pd.Series({
        "detected_lang": detected_lang,
        "nlp_topic": nlp_topic,
        "nlp_confidence": nlp_confidence,
   #     "gpt4_topic": gpt4_topic,
    #    "gpt4_confidence": gpt4_confidence
    })

# Apply the topic assignment function to the dataset
df_results = df.apply(assign_topics, axis=1)

# Merge the results with the original dataset
df_combined = pd.concat([df, df_results], axis=1)

# Save the results to a new CSV file
output_file = "ontolisst\ontolisst_results.csv"
df_combined.to_csv(output_file, index=False)
print(f"Results saved to {output_file}")

# Display a few samples from the annotated dataset
print("Annotated Samples:")
print(df_combined.head())