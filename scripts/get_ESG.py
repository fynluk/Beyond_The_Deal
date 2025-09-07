import lseg.data as ld
import pandas
import progressbar
import pandas as pd
import yaml
import logging
from IPython.display import display
from refinitiv.dataplatform import search
import refinitiv.data as rd
from concurrent.futures import ThreadPoolExecutor, as_completed

from scripts.get_ticker import get_tickers


def open_ld():
    try:
        ld.open_session()
        rd.open_session()
    except:
        logging.error("Refinitive Bridge connection failed")


def get_esg_scores(deals):
    target_rics = deals['TargetRIC'].tolist()
    acquirer_rics = deals['AcquirerRIC'].tolist()
    rics = target_rics + acquirer_rics
    print(len(rics))

    esg_data = pd.DataFrame()
    for ric in rics:
        if pd.isnull(ric):
            continue
        else:
            query = ld.get_data(
                universe=ric,
                fields=['TR.TRESGScore.fperiod', 'TR.TRESGScore'],
                parameters={
                    "Frq": "FY",
                    "SDate": "-20Y",
                    "EDate": "0Y",
                    "Period": "FY0"
                }
            )
            esg_data = pd.concat([esg_data, query], ignore_index=True)
    return esg_data


def get_deals():
    deal_filter = "TargetPublicStatus eq 'Public' and AcquirerPublicStatus eq 'Public' and (TransactionStatus eq 'Completed' or TransactionStatus eq 'Unconditional') and (TransactionAnnouncementDate le 2024-12-31 and TransactionAnnouncementDate ge 2020-01-01)"
    #deal_filter = "AcquirerCountry eq 'US'"
    #filter = FormOfTransactionName eq 'Acquisition' and
    df = ld.discovery.search(
        view=ld.discovery.Views.DEALS_MERGERS_AND_ACQUISITIONS,
        filter=deal_filter,
        #select= '_debugall',
        select='SdcDealNumber, TransactionAnnouncementDate, TargetCompanyName, TargetRIC, AcquirerCompanyName, AcquirerRIC, TransactionStatus, FormOfTransactionName',
        top= 10000,
        order_by= "TransactionAnnouncementDate"
    )
    #display(df.columns.values.tolist())
    #for key, value in df.iloc[0]['raw_source'].items():
    #    print(f"{key}: {value}")
    #print(df.iloc[0]['raw_source'])
    return df


def get_info(dfDeals):
    jointLists = dfDeals.copy()
    infos_list = []

    def process_row(index, row):
        print(f"{index}/{len(dfDeals)}")
        if pd.isnull(row['TargetRIC']):
            print(f"No Target-RIC for deal: {row['SdcDealNumber']}")
            return None
        elif pd.isnull(row['AcquirerRIC']):
            print(f"No Acquirer-RIC for deal: {row['SdcDealNumber']}")
            return None
        else:
            query = rd.get_data(
                universe=row["TargetRIC"],
                fields=[
                    'TR.MNASDCDealNumber',
                    'TR.MnADealValue(Scale=6)',
                    'TR.MnAEnterpriseValueAtAnnouncementDate(Scale=6)',
                    'TR.MnATargetEbitdaLTM(Scale=6)',
                    'TR.MnAEnterpriseValueToEBIDTA',
                    'TR.MnAEnterpriseValueToEBIT',
                    'TR.MnAPctOfSharesAcquired',
                    'TR.MnADealType'
                ]
            )
            return query

    # Multi-Threading
    with ThreadPoolExecutor(max_workers=8) as executor:  # max_workers anpassen
        futures = {executor.submit(process_row, idx, row): idx for idx, row in dfDeals.iterrows()}

        for future in as_completed(futures):
            idx = futures[future]
            try:
                result = future.result()
                if result is not None:
                    infos_list.append(result)
            except Exception as e:
                print(f"Error at index {idx}: {e}")

    # Ergebnisse zusammenführen
    if infos_list:
        infos = pd.concat(infos_list, ignore_index=True)
    else:
        infos = pd.DataFrame()

    # Merge mit Original
    fullInfo = pd.merge(jointLists, infos, left_on="SdcDealNumber", right_on="SDC Deal No", how="left")
    return fullInfo


def filterData(data):
    data['Deal Value'] = pd.to_numeric(data['Deal Value'], errors='coerce')
    filtered_data = data[data['M&A Type'] == 'Disclosed Dollar Value Deal']
    filtered_df = data.loc[(data['M&A Type'] == 'Disclosed Dollar Value Deal') & (data['Percentage of Shares Acquired in Transaction'] == 100) & (data['Deal Value'] >= 500)]

    return filtered_df


def main():
    logging.info("Open Refinitiv Session")
    open_ld()
    logging.info("Get Deals")
    deals = get_deals()
    display(deals.to_markdown())
    logging.info("Get Deal Information")
    data = get_info(deals)
    display(data.to_markdown())
    logging.info("Filter Query")
    filtered_data = filterData(data)
    filtered_data.to_excel("Deals3.xlsx", index=False)
    display(filtered_data.to_markdown())
    logging.info("Get ESG-Scores")
    esg_matrix = get_esg_scores(filtered_data)
    display(esg_matrix.to_markdown())
    esg_matrix.to_excel("ESG_Scores3.xlsx", index=False)
    
    
    #logging.info("Get ESG Score")
    #scores = get_esg_scores(['AAPL.O','IBM'])
    #print(scores)
    ld.close_session()


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, filename="../runtime.log", filemode="w",
                        format="%(asctime)s %(levelname)s %(message)s")
    pandas.set_option("future.no_silent_downcasting", True)
    main()