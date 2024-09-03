import pandas as pd
import random
import csv
import os
from tqdm import tqdm


unique_streets_df = pd.read_csv('resources/Street_Names (1).csv')
unique_cities_df = pd.read_csv('resources/unique_cities.csv')
cleaned_geonames_postal_code_df = pd.read_csv('resources/cleaned_geonames_postal_code.csv')

def generate_address():
    streets = unique_streets_df['street_name'].tolist()
    house_number = random.randint(1, 100)
    
    city_data = random.choice(cleaned_geonames_postal_code_df.values)
    postal_code = city_data[1]
    city = city_data[2]
    
    street_address = f"{random.choice(streets)} {house_number}"
    full_address = f"{street_address}, {postal_code} {city}"
    
    return full_address
def create_labeled_data_with_tokens(num_samples=100000, file_path="resources/labeled_data.csv"):
    start_templates = [
        "{address} is the address we are looking for.",
        "{address} is the location to visit.",
        "{address} is the location to get.",
        "{address} is where you should go.",
        "{address} is the correct location.",
        "{address} is the destination address.",
        "{address} is the place to find.",
        "{address} is where the event will be held.",
        "{address} is the address you need.",
        "{address} is the location of interest.",
        "{address} is the address mentioned."
    ]
    
    middle_templates = [
        "We have an important location at {address}.",
        "Please send the information to {address} as well.",
        "The correct address is {address}, located near the park.",
        "You can find the address {address} in the document.",
        "Refer to {address} for more details.",
        "Ensure the delivery is made to {address}.",
        "The office is located at {address} on the second floor.",
        "For assistance, visit {address} in the town center.",
        "Your destination, {address}, is on the right.",
        "The delivery should be made to {address}. Thank you!",
        "The new branch is at {address}, as per the latest update."
    ]
    
    end_templates = [
        "The correct address is {address}. Please confirm.",
        "You can reach us at {address}.",
        "The delivery should be made to {address}.",
        "All shipments are to be sent to {address}.",
        "Please address all correspondence to {address}.",
        "Send all inquiries to {address}.",
        "The address you need is {address}.",
        "Please make sure to visit {address}.",
        "Our office is located at {address}.",
        "The final destination is {address}."
    ]
    
    all_templates = start_templates + middle_templates + end_templates
    
    with open(file_path, mode='w', newline='', encoding='utf-8') as file:
        writer = csv.writer(file)
        writer.writerow(["text", "labels"])
        
        for i in tqdm(range(num_samples), desc="Processing samples"):
            address = generate_address()
            template = random.choice(all_templates)
            text = template.format(address=address)
            
            tokens = text.split()
            label = [0] * len(tokens)
            
            address_tokens = address.split()
            address_start = tokens.index(address_tokens[0])
            address_end = address_start + len(address_tokens)
            
            for j in range(address_start, address_end):
                label[j] = 1
            
            writer.writerow([text, ' '.join(map(str, label))])
    
    print(f"Data saved to {file_path}")
    
create_labeled_data_with_tokens(num_samples=100000, file_path="resources/labeled_data1.csv")