import json
import logging


def loggingFormat():
    logging.basicConfig(
        filename='predictions.log',
        level=logging.INFO,
        format='%(asctime)s - %(message)s'
    )


def naiveBayes(instance):
    with open('NaiveBayesModel.json', 'r') as json_file:
        model = json.load(json_file)
    likelihoods = model["likelihoods"]
    ppYes = model["priorProbabilities"]["Yes"]
    ppNo = model["priorProbabilities"]["No"]

    for attribute in instance:
        instanceValue = instance[attribute]
        ppYes *= likelihoods[attribute][instanceValue]["Yes"]
        ppNo *= likelihoods[attribute][instanceValue]["No"]
    if ppYes >= ppNo:
        return "Yes"
    return "No"


def alignTable(instanceData, width):
    tabLength = width - len(str(instanceData))
    tab = ""
    for i in range(tabLength):
        tab += " "
    return tab


def printConfusionMatrix(confusionMatrix):
    print("CONFUSION MATRIX")
    print("                    Positive          Negative")
    print("True               ", confusionMatrix['TruePositive'],
          alignTable(confusionMatrix['TruePositive'], 16), confusionMatrix['TrueNegative'])
    print("False              ", confusionMatrix['FalsePositive'],
          alignTable(confusionMatrix['FalsePositive'], 16), confusionMatrix['FalseNegative'])


def calculateConfusionMatrix(confusionMatrix, instance, prediction):
    if instance['PlayTennis'] == 'Yes' and prediction == 'Yes':
        confusionMatrix['TruePositive'] += 1
    elif instance['PlayTennis'] == 'No' and prediction == 'No':
        confusionMatrix['TrueNegative'] += 1
    elif instance['PlayTennis'] == 'Yes' and prediction == 'No':
        confusionMatrix['FalseNegative'] += 1
    elif instance['PlayTennis'] == 'No' and prediction == 'Yes':
        confusionMatrix['FalsePositive'] += 1
    return confusionMatrix


def evaluate(data):
    predictionCorrect = 0
    predictionIncorrect = 0
    confusionMatrix = {'TruePositive': 0, 'FalsePositive': 0, 'FalseNegative': 0, 'TrueNegative': 0}
    for instance in data:
        testInstance = instance.copy()
        testInstance.pop('Day')
        testInstance.pop('PlayTennis')
        prediction = naiveBayes(testInstance)

        logging.info(f"Instance: {testInstance}, Predicted Outcome: {prediction}, Actual Outcome: {instance['PlayTennis']}")

        if prediction == instance['PlayTennis']:
            predictionCorrect += 1
        else:
            predictionIncorrect += 1
        confusionMatrix = calculateConfusionMatrix(confusionMatrix, instance, prediction)
    accuracy = predictionCorrect / (predictionIncorrect + predictionCorrect) * 100

    print(f"ACCURACY\nAccuracy of the algorithm is {accuracy:.2f}\n")
    printConfusionMatrix(confusionMatrix)


def createModel(likelihoods, ppYes, ppNo):
    model = {
        "likelihoods": likelihoods,
        "priorProbabilities": {
            "Yes": ppYes,
            "No": ppNo
        }
    }
    with open('NaiveBayesModel.json', 'w') as json_file:
        json.dump(model, json_file, indent=4)


def laplaceSmoothing(likelihoods, table, yes, no):
    prevYes = yes + len(likelihoods[table])
    prevNo = no + len(likelihoods[table])
    for instance in likelihoods[table]:
        likelihoods[table][instance]["Yes"] = (likelihoods[table][instance]["Yes"] + 1) / prevYes
        likelihoods[table][instance]["No"] = (likelihoods[table][instance]["No"] + 1) / prevNo
    return likelihoods


def setProbability(likelihoods, table, yes, no):
    for instance in likelihoods[table]:
        likelihoods[table][instance]["Yes"] = likelihoods[table][instance]["Yes"] / yes
        likelihoods[table][instance]["No"] = likelihoods[table][instance]["No"] / no
    return likelihoods


def calculateLikelihoods(data, yes, no):
    likelihoods = {}
    attributes = []
    for attribute in data[0]:
        if attribute != "Day" and attribute != "PlayTennis":
            attributes.append(attribute)
            likelihoods[attribute] = {}

    for instance in data:
        for attribute in attributes:
            value = instance[attribute]
            if value not in likelihoods[attribute]:
                likelihoods[attribute][value] = {'Yes': 0, 'No': 0}
            if instance['PlayTennis'] == 'Yes':
                likelihoods[attribute][value]['Yes'] += 1
            else:
                likelihoods[attribute][value]['No'] += 1

    for table in likelihoods:
        zeroExists = False
        for instance in likelihoods[table]:
            for result in likelihoods[table][instance]:
                if likelihoods[table][instance][result] == 0:
                    likelihoods = laplaceSmoothing(likelihoods, table, yes, no)
                    zeroExists = True
                    break
            if zeroExists:
                break
        if not zeroExists:
            likelihoods = setProbability(likelihoods, table, yes, no)
    return likelihoods


def printData(data):
    print("Day  Outlook        Temperature    Humidity       Wind           PlayTennis")
    for instance in data:
        print(instance['Day'], alignTable(instance['Day'], 3),
              instance['Outlook'], alignTable(instance['Outlook'], 13),
              instance['Temperature'], alignTable(instance['Temperature'], 13),
              instance['Humidity'], alignTable(instance['Humidity'], 13),
              instance['Wind'], alignTable(instance['Wind'], 13),
              instance['PlayTennis'], alignTable(instance['PlayTennis'], 13))


def prepareData():
    with open('PlayTennisData.json', 'r') as json_file:
        data = json.load(json_file)
    printData(data)
    return data


def main():
    loggingFormat()

    data = prepareData()

    yes = 0
    no = 0

    for instance in data:
        if instance['PlayTennis'] == 'Yes':
            yes += 1
        elif instance['PlayTennis'] == 'No':
            no += 1

    print("\nNumber of Yes:", str(yes))
    print("Number of No:", str(no) + "\n")

    ppYes = yes / (yes + no)
    ppNo = no / (yes + no)
    likelihoods = calculateLikelihoods(data, yes, no)

    createModel(likelihoods, ppYes, ppNo)
    evaluate(data)


if __name__ == "__main__":
    main()
