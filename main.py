import json


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


def alignTable(instanceData, width):
    tabLength = width - len(str(instanceData))
    tab = ""
    for i in range(tabLength):
        tab += " "
    return tab


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

    newInstance = {'Outlook': 'Rain', 'Temperature': 'Cool', 'Humidity': 'Normal', 'Wind': 'Strong'}
    prediction = naiveBayes(newInstance)
    print("The predicted PlayTennis result for the instance", newInstance, "is", prediction)


if __name__ == "__main__":
    main()
