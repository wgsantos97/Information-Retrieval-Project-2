import json

population = 0  # Overall population
noNames = 0  # Keeps track of entries that are missing names
total = 0
average = 0
longestName = ""
longest = 0
shortestName = ""
shortest = 99999


def stats(dictList):
    global total
    global average
    global longest
    global longestName
    global shortest
    global shortestName

    idx = 0
    print("Generating Statistics: ")
    for d in dictList:
        idx = idx + 1
        textLength = len(d["review_text"].split())
        total = total + textLength
        if(textLength > longest):
            longestName = d["title"] + " [ID #" + d["book_id"] + "]"
            longest = textLength
        if(textLength < shortest):
            shortestName = d["title"] + " [ID #" + d["book_id"] + "]"
            shortest = textLength
        print("Progress: " + str(round(idx/population*100, 5)) + "%")
    average = round(total/population)


def main():
    global noNames
    global population

    revData = "BaseData/graphic_novel_reviews.json"
    namData = "BaseData/graphic_novels.json"
    finData = "graphic_novel_final.json"
    results = "results.md"

    # STEP 1: Assimilate the 2 json files into a list of dictionary objects
    dictList = list()  # List of dictionary entries
    noNamesList = list()

    # STEP 1a: Write the graphic_novels.json file to a dictionary
    f = open(namData, 'r')
    nameDict = json.load(f)
    f.close()

    # STEP 1b: Open the graphic_novel_reviews.json file and cross-reference it
    # against the nameDict dictionary. Use this to synthesize a new dictionary
    # entry to the dictList that has all the desired field entries.
    f = open(revData, 'r')
    print("Merging " + revData + " & " + namData + " json files now...")
    for line in f:
        d = json.loads(line)
        population = population + 1  # population report

        n = "n/a"  # default
        if(d["book_id"] in nameDict):  # Attach book_name to corresponding id number
            n = nameDict[d["book_id"]]
        else:
            noNames = noNames + 1
            noNamesList.append(entry)

        entry = {
            "book_id": d["book_id"],
            "title": n,
            "rating": d["rating"],
            "review_text": d["review_text"],
        }
        dictList.append(entry)
    f.close()

    # STEP 2: Write to new json file
    print("Writing new json data to " + finData + "...")
    f = open(finData, 'w')
    for data in dictList:
        json.dump(data, f)
        f.write("\n")
    f.close()

    # STEP 3: Write the report
    print("Writing results to " + results + "...")
    stats(dictList)
    f = open(results, 'w')
    f.write("# JSON Data Summary\n\n")

    f.write("## General Stats\n\n")
    f.write("\tAverage: " + str(average) + " words\n")
    f.write("\tPopulation: " + str(population) + " entries\n")
    f.write("\tLongest Entry: " + str(longestName) +
            " - " + str(longest) + " words \n")
    f.write("\tShortest Entry: " + str(shortestName) +
            " - " + str(shortest) + " words \n")

    f.write("\n## No Name Entries\n\n")
    f.write("\tPopulation: " + str(noNames) + "\n")
    f.write("\tList: \n")
    idx = 1
    if(len(noNamesList) == 0):
        f.write("\t\tNo Entries Found\n")
    for data in noNamesList:
        f.write("\t\t" + str(idx) + ") " + data["book_id"])
        idx = idx+1
    f.close()


main()
