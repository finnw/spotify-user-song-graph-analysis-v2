import csv

class LabelEncoding:
    """Assign unique sequential numeric ids to arbitrary values."""
    def __init__(self):
        self.mapping = {}
        self.reverse_mapping = None
        self.repeated_values = set()

    def get_or_create_id(self, value):
        """Return the numeric id for value, assigning a new sequential id if necessary."""
        if value in self.mapping:
            id = self.mapping[value]
            self.repeated_values.add(id)
        else:
            id = self.mapping[value] = len(self.mapping)
        return id

    def get_value(self, id):
        """Return the value corresponding to the given numeric id."""
        if self.reverse_mapping is None:
            self.reverse_mapping = {v: k for k, v in self.mapping.items()}
        return self.reverse_mapping[id]

    def export_csv(self, file_path, heading = None):
        """Export the mapping to a CSV file."""
        with open(file_path, 'w', newline='', encoding='utf-8') as csvfile:
            writer = csv.writer(csvfile)
            if heading:
                writer.writerow(heading)
                if len(heading) == 2:
                    for value, id in self.mapping.items():
                        writer.writerow([id, value])
                else:    
                    for value, id in self.mapping.items():
                        writer.writerow(list((id,) + tuple(value)))
            else:
                for value, id in self.mapping.items():
                    writer.writerow([id, value])
    
    def import_csv(self, file_path):
        """Import the mapping from a CSV file."""
        with open(file_path, 'r', encoding='utf-8') as csvfile:
            reader = csv.reader(csvfile)
            next(reader)  # Skip header
            for row in reader:
                id, value = row
                self.mapping[value] = int(id)
        self.reverse_mapping = None  # Reset reverse mapping