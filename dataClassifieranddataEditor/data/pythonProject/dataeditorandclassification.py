import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.metrics import accuracy_score
from sklearn.dummy import DummyClassifier

class DataEdit:
    def __init__(self, file_path, encoding=None):
        self.data = pd.read_csv(file_path, encoding=encoding)

    def drop_columns(self, columns_to_drop):
        self.data = self.data.drop(columns=columns_to_drop)

    def drop_na(self):
        self.data = self.data.dropna()

    def head(self, rows):
        self.data = self.data.head(rows)

    def save(self, file_path, index=False):
        self.data.to_csv(file_path, index=index)

    def print_first_50(self):
        print(self.data.head(50))

class DataCombining:
    def __init__(self, file_path1, file_path2, encoding=None):
        self.data1 = pd.read_csv(file_path1, encoding=encoding)
        self.data2 = pd.read_csv(file_path2, encoding=encoding)

    def combine_data(self):
        combined_data = pd.concat([self.data1, self.data2], axis=0)
        return combined_data

    def save_combined_data(self, file_path, index=False):
        combined_data = self.combine_data()
        combined_data.to_csv(file_path, index=index)

def main():
    while True:
        print("\nHangi işlemi yapmak istersiniz?")
        print("1. Veri Düzenleme")
        print("2. Veri Birleştirme")
        print("3. Sınıflandırma Algoritmaları")
        print("0. Çıkış")
        choice = input("Seçiminizi yapın (0/1/2/3): ")

        if choice == "0":
            print("Programdan çıkılıyor...")
            break

        elif choice == "1":
            file_path = input("CSV dosyasının tam yolunu girin (örneğin: C:/Users/Osman/Desktop/test1.csv)\n filepath: ")
            edit = DataEdit(file_path, encoding='latin1')

            print("\nDüşürmek istediğiniz sütunları virgülle ayırarak girin (örneğin: No.,Source,Length,Info,Destination)")
            print(f"Veri setindeki sütunlar: {list(edit.data.columns)}")
            columns_to_drop = input("Düşürmek istediğiniz sütunlar: ")
            edit.drop_columns(columns_to_drop.split(","))

            rows = int(input("\nBaştan kaç satır veriyi kullanmak istersiniz: "))
            edit.head(rows)

            save_path = input("\nKaydedilecek dosyanın yolunu girin (örneğin: C:/Users/Osman/Desktop/test1Edited.csv)\n filepath: ")
            edit.save(save_path)

            print("\nVeri düzenleme işlemi başarıyla tamamlandı.")

        elif choice == "2":
            file_path1 = input("İlk CSV dosyasının yolunu girin (örneğin: C:/Users/Osman/Desktop/test1Edited.csv) \n filepath: ")
            file_path2 = input("İkinci CSV dosyasının yolunu girin (örneğin: C:/Users/Osman/Desktop/test2Edited.csv) \n filepath: ")
            combine = DataCombining(file_path1, file_path2, encoding='latin1')

            save_path = input("Kaydedilecek dosyanın yolunu girin (örneğin: C:/Users/Osman/Desktop/test3Combined.csv)\n filepath: ")
            combine.save_combined_data(save_path)

            print("\nVeri birleştirme işlemi başarıyla tamamlandı.")

        elif choice == "3":
            file_path = input("Sınıflandırma yapılacak CSV dosyasının tam yolunu girin (örneğin: C:/Users/Osman/Desktop/5combined_data.csv)\n filepath: ")
            data = pd.read_csv(file_path)

            # Özellikler ve etiketleri ayırma
            X = data.drop(columns=["Protocol"])
            y = data["Protocol"]

            # Eğitim ve test verisi ayırma
            X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

            while True:
                print("\nKullanmak istediğiniz sınıflandırma algoritmasını seçin:")
                print("a. RandomForestClassifier")
                print("b. DecisionTreeClassifier")
                print("c. GaussianNB (Naive Bayes)")
                print("d. ZeroRClassifier")
                print("0. Çıkış")

                sub_choice = input("Seçiminizi yapın (0/a/b/c/d): ")

                if sub_choice == '0':
                    print("Algoritma seçiminden çıkılıyor.")
                    break

                elif sub_choice == 'a':
                    # RandomForest sınıflandırıcıyı oluşturma ve eğitme
                    rf = RandomForestClassifier(random_state=42)
                    rf.fit(X_train, y_train)

                    # Test verileri üzerinde tahmin yapma
                    y_pred = rf.predict(X_test)

                elif sub_choice == 'b':
                    # Decision Tree sınıflandırıcıyı oluşturma ve eğitme
                    dt = DecisionTreeClassifier(random_state=42)
                    dt.fit(X_train, y_train)

                    # Test verileri üzerinde tahmin yapma
                    y_pred = dt.predict(X_test)

                elif sub_choice == 'c':
                    # Naive Bayes sınıflandırıcıyı oluşturma ve eğitme
                    nb = GaussianNB()
                    nb.fit(X_train, y_train)

                    # Test verileri üzerinde tahmin yapma
                    y_pred = nb.predict(X_test)

                elif sub_choice == 'd':
                    # ZeroR sınıflandırıcıyı oluşturma ve eğitme
                    zero_r = DummyClassifier(strategy="most_frequent")
                    zero_r.fit(X_train, y_train)

                    # Test verileri üzerinde tahmin yapma
                    y_pred = zero_r.predict(X_test)

                else:
                    print("Geçersiz seçim. Lütfen tekrar deneyin.")
                    continue

                # Başarı oranını hesaplama ve çıktı olarak yazdırma
                accuracy = accuracy_score(y_test, y_pred)
                print("Başarı Oranı:", accuracy)

        else:
            print("\nGeçersiz seçim. Lütfen tekrar deneyin.")

if __name__ == "__main__":
    main()
