import src.prepare_data as prepare_data
import src.train_classifier as train_classifier
import src.detect_objects as detect_objects

if __name__ == "__main__":
    # Przygotowanie danych
    prepare_data.create_positives_file('dataset/positives', 'dataset/positives.txt')
    prepare_data.create_negatives_file('dataset/negatives', 'dataset/negatives.txt')

    # Generowanie wektorów cech (zakładając, że używasz narzędzia opencv_createsamples poza Pythonem)
    os.system('opencv_createsamples -info dataset/positives.txt -num 1000 -w 24 -h 24 -vec dataset/positives.vec')

    # Trenowanie klasyfikatora
    train_classifier.train_classifier('dataset/positives.vec', 'dataset/negatives.txt', 'haar-cascade', 900, 500, 10)

    # Wykrywanie obiektów
    detect_objects.detect_objects('test.jpg', 'haar-cascade/cascade.xml')
