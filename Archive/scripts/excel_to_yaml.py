

ctr = 0
excel_filename = "input.csv"
yaml_filename = "output.yaml"
deals = {}
counter = 0

with open(excel_filename, "r") as excel_csv:
    for line in excel_csv:
        if counter == 0:
            counter = counter + 1  # Skip the header
        else:
            # save the csv as a dictionary
            aDate,target_name,target_ticker,buyer_name,buyer_ticker = line.replace(' ','').strip().split(';', maxsplit=4)

            # Prepare Date
            dd,mm,yy = aDate.split('.')
            date = '20'+yy+'-'+mm+'-'+dd

            deals[counter-1] = {'buyer': buyer_ticker, 'target': target_ticker, 'aDate': date}
            print(deals[counter - 1])
            counter = counter + 1

with open(yaml_filename, "w+") as yf :
    yf.write("Deals: \n")
    for deal in deals:
        for k,v in deals[deal].items():
            if k == 'buyer':
                yf.write(f"  - {k}: {v}\n")
            else:
                yf.write(f"    {k}: {v}\n")