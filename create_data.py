import pandas as pd
import random
import csv

from tqdm import tqdm

unique_streets_df = pd.read_csv('resources/Street_Names (1).csv')
unique_cities_df = pd.read_csv('resources/unique_cities.csv')
cleaned_geonames_postal_code_df = pd.read_csv('resources/cleaned_geonames_postal_code.csv')
unique_names_df = pd.read_csv('resources/unique_names.csv', header=None, names=['name'])
unique_companies_df = pd.read_csv('resources/German_companies_names.csv')

german_english_floors = [
    "3. OG", "App. 4", "1. Stock", "2. OG", "4. Stock", "EG", "Dachgeschoss", 
    "Keller", "Souterrain", "Penthouse", "5. OG", "App. 12", "DG", "UG", 
    "3. Stock rechts", "4. OG links", "3rd Floor", "Apt. 4", "1st Floor", "2nd Floor", "4th Floor", "Ground Floor", 
    "Attic", "Basement", "Lower Ground", "Penthouse", "5th Floor", "Apt. 12", 
    "Top Floor", "Underground Floor", "3rd Floor Right", "4th Floor Left"
]
prefixes = [
    "Herr", "Frau", "Dr.", "Prof.", "Herr Dr.", "Frau Dr.", 
    "Herr Prof.", "Frau Prof.", "Herr Dipl.-Ing.", "Frau Dipl.-Ing.",
    "Herr Mag.", "Frau Mag.", "Herr Ing.", "Frau Ing.", "Mr.", "Ms.",
    "Mrs.", "Mx.", "Sir", "Madam", "Lord", "Lady"
]
countries = [
    "Afghanistan", "Albania", "Algeria", "Andorra", "Angola", "Antigua and Barbuda", "Argentina", 
    "Armenia", "Australia", "Austria", "Azerbaijan", "Bahamas", "Bahrain", "Bangladesh", "Barbados", 
    "Belarus", "Belgium", "Belize", "Benin", "Bhutan", "Bolivia", "Bosnia and Herzegovina", "Botswana", 
    "Brazil", "Brunei", "Bulgaria", "Burkina Faso", "Burundi", "Cabo Verde", "Cambodia", "Cameroon", 
    "Canada", "Central African Republic", "Chad", "Chile", "China", "Colombia", "Comoros", "Congo", 
    "Costa Rica", "Croatia", "Cuba", "Cyprus", "Czech Republic", "Denmark", "Djibouti", "Dominica", 
    "Dominican Republic", "Ecuador", "Egypt", "El Salvador", "Equatorial Guinea", "Eritrea", "Estonia", 
    "Eswatini", "Ethiopia", "Fiji", "Finland", "France", "Gabon", "Gambia", "Georgia", "Germany", 
    "Ghana", "Greece", "Grenada", "Guatemala", "Guinea", "Guinea-Bissau", "Guyana", "Haiti", "Honduras", 
    "Hungary", "Iceland", "India", "Indonesia", "Iran", "Iraq", "Ireland", "Israel", "Italy", "Jamaica", 
    "Japan", "Jordan", "Kazakhstan", "Kenya", "Kiribati", "Kuwait", "Kyrgyzstan", "Laos", "Latvia", 
    "Lebanon", "Lesotho", "Liberia", "Libya", "Liechtenstein", "Lithuania", "Luxembourg", "Madagascar", 
    "Malawi", "Malaysia", "Maldives", "Mali", "Malta", "Marshall Islands", "Mauritania", "Mauritius", 
    "Mexico", "Micronesia", "Moldova", "Monaco", "Mongolia", "Montenegro", "Morocco", "Mozambique", 
    "Myanmar", "Namibia", "Nauru", "Nepal", "Netherlands", "New Zealand", "Nicaragua", "Niger", "Nigeria", 
    "North Macedonia", "Norway", "Oman", "Pakistan", "Palau", "Palestine", "Panama", "Papua New Guinea", 
    "Paraguay", "Peru", "Philippines", "Poland", "Portugal", "Qatar", "Romania", "Russia", "Rwanda", 
    "Saint Kitts and Nevis", "Saint Lucia", "Saint Vincent and the Grenadines", "Samoa", "San Marino", 
    "Sao Tome and Principe", "Saudi Arabia", "Senegal", "Serbia", "Seychelles", "Sierra Leone", "Singapore", 
    "Slovakia", "Slovenia", "Solomon Islands", "Somalia", "South Africa", "South Sudan", "Spain", "Sri Lanka", 
    "Sudan", "Suriname", "Sweden", "Switzerland", "Syria", "Tajikistan", "Tanzania", "Thailand", "Timor-Leste", 
    "Togo", "Tonga", "Trinidad and Tobago", "Tunisia", "Turkey", "Turkmenistan", "Tuvalu", "Uganda", "Ukraine", 
    "United Arab Emirates", "United Kingdom", "United States", "Uruguay", "Uzbekistan", "Vanuatu", "Vatican City", 
    "Venezuela", "Vietnam", "Yemen", "Zambia", "Zimbabwe", "Deutschland"
]

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
        
        house_number = str(random.randint(1, 100)) if random.random() < 0.5 else str(random.randint(1, 10))
            
        if random.random() < 0.4:
            house_number += random.choice('aaaabbbbccccdddefg')
        floor_app = random.choice(german_english_floors) # ToDo:
        
        city_data = random.choice(cleaned_geonames_postal_code_df.values)
        postal_code = str(city_data[1])
        city = city_data[2]
        
        street_address = f"{random.choice(streets)} {house_number} {floor_app}"
        return street_address, postal_code, city

    @staticmethod
    def generate_name():
        try:
            prefix = ""
            if random.random() < 0.3:
                prefix = random.choice(prefixes)
            name = random.choice(unique_names_df['name'].tolist())
            surname = random.choice(unique_names_df['name'].tolist())

            if isinstance(name, str):
                name = name[:1].upper() + name[1:].lower()
            else:
                raise ValueError("Invalid type for name")

            if isinstance(surname, str):
                surname = surname[:1].upper() + surname[1:].lower()
            else:
                raise ValueError("Invalid type for surname")

        except Exception as e:
            print(f"Error: {e}")
            return "Max Verstappen"

        full_name = f"{prefix} {name} {surname}" if prefix != "" else f"{name} {surname}"
            
        return full_name
    
        
    @classmethod
    def generate_full_address(cls):
        name = cls.generate_name()
        company = random.choice(unique_companies_df['name'].tolist())

        position = random.choice([
            "Abteilung Vertrieb", "Manager", "Technician", "Developer", "Projektleiter", 
            "Analyst", "Consultant", "Sales Representative", "Teamleiter", "Ingenieur", 
            "CEO", "CTO", "CFO", "Product Manager", "Support Specialist", "Marketing Coordinator",
            "Researcher", "HR Manager", "Operations Director", "Data Scientist"
        ]) # todo
        
        country = random.choice(countries)
        street_address, postal_code, city = cls.generate_address()

        return cls(name, company, position, country, street_address, postal_code, city)

    def get_address_formats(self):
        possible_formats = [
            f"{self.name}\n{self.street_address}\n{self.postal_code} {self.city}\n{self.country}",
            f"{self.name}\n{self.street_address}\n{self.postal_code} {self.city}",
            f"{self.company}\n{self.name}\n{self.street_address}\n{self.postal_code} {self.city}\n{self.country}",
            f"{self.company}\n{self.name}\n{self.street_address}\n{self.postal_code} {self.city}",
            f"{self.company}\n{self.position} {self.name}\n{self.street_address}\n{self.postal_code} {self.city}\n{self.country}",
            f"{self.company}\n{self.position}\n{self.street_address}\n{self.postal_code} {self.city}",
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

create_labeled_data_with_tokens(num_samples=150000, file_path="resources/test_labeled_data.csv")
