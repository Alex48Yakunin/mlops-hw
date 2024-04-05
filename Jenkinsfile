pipeline {
    agent any
    stages {
        stage('Запуск скрипта создания данных') {
            steps {
                python data_creation.py
            }
        }
        stage('Запуск скрипта предобработки данных') {
            steps {
                python model_preprocessing.py
            }
        }
        stage('Запуск скрипта подготовки и обучения модели') {
            steps {
                python model_preparation.py
            }
        }
        stage('Запуск скрипта тестирования модели') {
            steps {
                python model_testing.py
            }
        }
    }
}
