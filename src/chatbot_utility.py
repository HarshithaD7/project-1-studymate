import os

PROJECT_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

def get_chapter_list(selected_class, selected_subject='Biology'):
    if not selected_class or not selected_subject:
        return []
    class_folder = selected_class.lower().replace(' ', '_')
    folder = os.path.join(PROJECT_DIR, 'data', class_folder, selected_subject.lower())
    if not os.path.exists(folder):
        return []
    chapters = [f[:-4] for f in os.listdir(folder) if f.lower().endswith('.pdf')]
    def sort_key(name):
        first = name.split('.')[0].split('_')[0].strip()
        try:
            return (0, int(first))
        except ValueError:
            return (1, name.lower())
    return sorted(chapters, key=sort_key)
