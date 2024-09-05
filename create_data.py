import pandas as pd
import random
import csv

from tqdm import tqdm

unique_streets_df = pd.read_csv('resources/Street_Names (1).csv')
unique_cities_df = pd.read_csv('resources/unique_cities.csv')
cleaned_geonames_postal_code_df = pd.read_csv('resources/cleaned_geonames_postal_code.csv')
unique_names_df = pd.read_csv('resources/unique_names.csv', header=None, names=['name'])

class Address:
    def __init__(self, name, company, position, country, street_address, postal_code, city):
        self.name = name
        self.company = company
        self.position = position
        self.country = country
        self.street_address = street_address
        self.postal_code = postal_code
        self.city = city
    
    @staticmethod
    def generate_address():
        streets = unique_streets_df['street_name'].tolist()
        house_number = random.randint(1, 100)
        floor_app = random.choice(["3. OG", "App. 4", "1. Stock"]) # todo
        
        city_data = random.choice(cleaned_geonames_postal_code_df.values)
        postal_code = str(city_data[1])
        city = city_data[2]
        
        street_address = f"{random.choice(streets)} {house_number} {floor_app}"
        return street_address, postal_code, city

    @staticmethod
    def generate_name():
        name = random.choice(unique_names_df['name'].tolist())
        name = name[0] + name[1:].lower()
        surname = random.choice(unique_names_df['name'].tolist())
        surname = surname[0] + surname[1:].lower()
        full_name = f"{name} {surname}"
        return full_name
    
        
    @classmethod
    def generate_full_address(cls):
        name = cls.generate_name()
        company = random.choice(["Musterfirma GmbH", "Tech Solutions AG", "Business Inc."])
        position = random.choice(["Abteilung Vertrieb", "Manager", "Technician", "Developer"]) # todo
        
        country = random.choice(["Germany", "France", "Ukraine"]) # todo
        street_address, postal_code, city = cls.generate_address()

        return cls(name, company, position, country, street_address, postal_code, city)

    def get_address_formats(self):
        possible_formats = [
            f"{self.name}\n{self.street_address}\n{self.postal_code} {self.city}\n{self.country}",
            f"{self.name}\n{self.street_address}\n{self.postal_code} {self.city}",
            f"{self.company}\n{self.name}\n{self.street_address}\n{self.postal_code} {self.city}\n{self.country}",
            f"{self.company}\n{self.name}\n{self.street_address}\n{self.postal_code} {self.city}",
            f"{self.name}\n{self.position}\n{self.street_address}\n{self.postal_code} {self.city}\n{self.country}",
            f"{self.name}\n{self.position}\n{self.street_address}\n{self.postal_code} {self.city}",
            f"{self.name}\n{self.street_address}\n{self.postal_code} {self.city}\n{self.country}",
            f"{self.name}\n{self.street_address}\n{self.postal_code} {self.city}",
        ]
        return possible_formats

    def get_random_format(self):
        return random.choice(self.get_address_formats())

def create_labeled_data_with_tokens(num_samples=100000, file_path="resources/labeled_data1.csv"):
    with open(file_path, mode='w', newline='', encoding='utf-8') as file:
        writer = csv.writer(file)
        writer.writerow(["text", "labels"])

        for _ in tqdm(range(num_samples), desc="Processing samples"):
            address_obj = Address.generate_full_address()
            full_address = address_obj.get_random_format()
            full_address = full_address.replace('{', '{{').replace('}', '}}')
            template = "{address}"
            text = template.format(address=full_address)
            
            tokens = text.split()
            labels = ['OOV'] * len(tokens)

            for idx, token in enumerate(tokens):
                if token in address_obj.name.split():
                    labels[idx] = 'PER'
                elif token in address_obj.company.split():
                    labels[idx] = 'COMP'
                elif token in address_obj.position.split():
                    labels[idx] = 'POS'
                elif token == address_obj.postal_code:
                    labels[idx] = 'ZIP'
                elif token in address_obj.street_address.split():
                    labels[idx] = 'STREET'
                elif token in address_obj.city.split():
                    labels[idx] = 'CITY'
                elif token in address_obj.country.split():
                    labels[idx] = 'COUNTRY'
            writer.writerow([text, ' '.join(labels)])

    print(f"Data saved to {file_path}")

# Вызов функции
create_labeled_data_with_tokens(num_samples=1000, file_path="resources/test_labeled_data.csv")
