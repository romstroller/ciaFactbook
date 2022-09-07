from FileTools import FileTools  # see github.com/romstroller/FileTools
from matplotlib import pyplot as plt
import pandas as pd
import numpy as np
import pickle
import math
import re
import os


fTools = FileTools()


def excludeParenth( _mtchLi, _val ):
    mtchLi = [ ]
    # # Drop value if between parentheses (pre-match open count > close count)
    for match in _mtchLi:
        matchDex = _val.index( match )
        openCount = _val[ :matchDex ].count( '(' )
        closCount = _val[ :matchDex ].count( ')' )
        parenthesised = openCount > closCount
        if not parenthesised: mtchLi.append( match )
    
    return mtchLi


def getMatchRemain( df_in, coIdex, patrn ):
    # take number-pattern match and save remainder for unit and scale
    mtches = df_in.iloc[ :, coIdex ].str.findall( patrn )
    
    pos = 0
    rmnder = [ ]
    mtches_ret = [ ]
    
    for roVal in df_in.iloc[ :, coIdex ]:
        matchLi = mtches[ pos ]
        if type( matchLi ) != list: rmnt = None  # is float; no remain
        elif len( matchLi ) == 1: rmnt = roVal
        # if more than one, truncate remainder before second match
        elif len( matchLi ) > 1: rmnt = roVal[ :roVal.index( matchLi[ 1 ] ) ]
        else: rmnt = None
        
        if rmnt:
            # exclude any parenthesised matches
            matchLi = excludeParenth( mtches[ pos ], roVal )
            mtches_ret.append( matchLi )
            if len( matchLi ) > 0:
                rmnder.append( rmnt.replace( matchLi[ 0 ], '' ) )
        else:
            if type( matchLi ) == float: mtches_ret.append( matchLi )
            else: mtches_ret.append( np.nan )
            rmnder.append( "" )
        
        pos += 1
    
    return mtches_ret, rmnder


# REGEX:
#   capture group             (
#   zero/one                  [+-]?             possible number sign
#   1-3 nums                  \d{1,3}           up to three straight nums
#   non-capture subgroup      (?:               possible thousand-groups
#     comma and 3 nums          ,\d{3}          (sep. comma)
#     zero or more times        )*
#   non-capture subgroup      (?:               then possible decimal part
#     decimal and 1+nums        \.\d+
#     zero/one time             )?
#   OR (alt. to last seq)     |                 or no groups, just
#     0+ nums, dec, 1+nums      \d*\.\d+        more nums and poss decimal
#   OR (alt. to last seq)     |
#     1+ nums                   \d+             or just more numbers.
#   Close capture group       )
#   ( only match "plain" number format last to capture complex num segments )

patt = re.compile( r'([+-]?\d{1,3}(?:,\d{3})*(?:\.\d+)?|\d*\.\d+|\d+)' )


def generateMatchDct(dffBook):
    
    dfFbDict = { }  # collect column data
    
    colDex = 1
    for colName in dffBook.columns[ colDex: ]:
        origCol = dffBook.iloc[ :, colDex ]
        colType = dffBook[ colName ].dtype
        
        # get match if string, store if already float, catch unexpected
        if colType == float:
            colDict = { 'matchedNums': origCol, 'remainder': [ ] }
        elif colType == np.float64:
            colDict = { 'matchedNums': origCol.astype( float ), 'remainder': [ ] }
        else:
            matches, remainder = getMatchRemain( dffBook, colDex, patt )
            colDict = { 'matchedNums': matches, 'remainder': remainder }
        
        colDict[ 'origCol' ] = origCol
        dfFbDict[ colName ] = colDict
        colDex += 1
        
    return dfFbDict


def sortMatches( matchList ):
    # take first match item as float to "clean", store else
    firstVals, splitVals, checkVals = [ ], [ ], [ ]
    for mNum in range( len( matchList ) ):
        el = matchList[ mNum ]
        isFilldList = (type( el ) == list) and (len( el ) > 0)
        if isFilldList:  # remove any thousandcomma to support convert
            firstVals.append( float( ''.join( el[ 0 ].split( ',' ) ) ) )
            splitVals.append( [ v for v in el[ 1: ] ] )
        elif type( el ) == np.float64:
            firstVals.append( float(el) )
        else:  # check all else are either nan or empty matchlist
            if ((type( el ) == list and len( el ) > 0) or
                (type( el ) != list and math.isnan( el ) is False)):
                checkVals.append( el )
            firstVals.append( np.nan )
            splitVals.append( np.nan )
    return firstVals, splitVals, checkVals


def isolateClean(_dfFbDict):
    # apply split-sort to match records
    for col in _dfFbDict:
        colDict = _dfFbDict[ col ]
        colDict[ 'clean' ], colDict[ 'splitVals' ], colDict[ 'checkVals' ] = (
            sortMatches( colDict[ 'matchedNums' ] ))
        
        # Report uncategorized data (eg val for each different type)
        if len( colDict[ 'checkVals' ] ) > 0:
            print( f"Got checkvals for {col}:" )
            typeSet = set( type( v ) for v in colDict[ 'checkVals' ] )
            egTVals = [ [ v, t ] for v, t in zip(
                typeSet, colDict[ 'checkVals' ] ) ]
            for t, v in egTVals: print( f"[{v}] is [{t}]\n" )


def getCleanDF(_dfFbDict, _df):
    # dictionary columns to DF, checking is now float
    newCols = [ ]
    cleanDF = _df.iloc[ :, 0 ]  # start with countries
    for col in _dfFbDict:
        clean = pd.Series( _dfFbDict[ col ][ 'clean' ] )
        lenFloat = len( [ i for i in clean if type( i ) == float ] )
        if lenFloat > len( clean ) * 0.90:
            newCols.append( col )
            cleanDF = pd.concat( [ cleanDF, clean ], axis=1 )
        else: print( "col is less than 90% float. Dropping..." )
    
    cleanDF.columns = [ 'Country' ] + newCols
    return cleanDF


def nanThreshold( notNan ):  # average plus .5 standard deviation (rounded)
    return int( (sum( notNan ) / len( notNan )) + 0.5 * np.std( notNan ) )


def nonNanFromDims( dfr, dim = 1 ):
    # Enforce non-nan threshold for dimensions
    nonNans = [ ]
    for pos in range( 0, dfr.shape[ dim ] ):
        if dim == 1: vals = dfr.iloc[ :, pos ].tolist()
        else: vals = dfr.loc[ [ pos ] ].values.tolist()[ 0 ]
        
        nonNans.append( [ vals, len( [ v for v in vals if
            type( v ) == float and not math.isnan( v ) ] ) ] )
    
    _thresh = nanThreshold( [ nval for _, nval in nonNans ] )
    keepVals = [ kval for kval, nnul in nonNans if nnul >= _thresh ]
    print( f"non-nan[ {len( keepVals )} ] thr[ {_thresh} ] dim[ {dim} ]" )
    
    return keepVals


# DROP ROWS - disabled to keep all significant countries
def cleanRows(dfFloat):
    dfClean = pd.DataFrame( nonNanFromDims( dfFloat, dim=0 ) )
    # add a columnindex row to track names of kept columns
    dfClean.loc[ -1 ] = dfFloat.columns
    dfClean.index = dfClean.index + 1
    dfClean.sort_index( inplace=True )
    return dfClean


# dfRowsClean = cleanRows()
# keepCols_RC = nonNanFromDims( dfRowsClean, dim=1 )
# # convert to numeric df - row-cleaned
# dfRCC = pd.DataFrame( { col[ 0 ]: col[ 1: ] for col in keepCols_RC } )
# dfRCC = dfRCC.apply( pd.to_numeric, errors='ignore' )
# dfColsClean = dfRCC


def getNumericNonNan(_df):
    # Enforce non-nan threshold for dimensions
    #   ( av. density plus .5 standard deviation (rounded) ),
    # convert to numeric, adding featname row for tracking through clean
    _df.loc[ -1 ] = _df.columns
    _df.index = _df.index + 1
    _df.sort_index( inplace=True )
    keepCols = nonNanFromDims( _df, dim=1 )
    dfNumClean = pd.DataFrame( { col[ 0 ]: col[ 1: ] for col in keepCols } )
    dfNumClean = dfNumClean.apply( pd.to_numeric, errors='ignore' )
    dfNumClean.insert( 0, 'Country', _df.iloc[ :, 0 ].tolist()[ 1: ] )
    return dfNumClean


def cleanReport(dfOrig, dfClean ):
    fbIsNa = dfOrig.isna().sum().sum()
    dfIsNa = dfClean.isna().sum().sum()
    fbDim = dfOrig.shape[ 0 ] * dfOrig.shape[ 1 ]
    dfDim = dfClean.shape[ 0 ] * dfClean.shape[ 1 ]
    return (
        f"factbook originally shape: {dfOrig.shape}\n"
        f"    NAN-density: {(fbIsNa / fbDim):.2f}% "
        f"({fbIsNa} NaN in {fbDim})\n"
        f"clean dataframe has shape: {dfClean.shape}\n"
        f"    NAN-density: {(dfIsNa / dfDim):.2f}% "
        f"({dfIsNa} NaN in {dfDim})\n" )


def runScaleAnalysis( dfr, remDict ):
    colList = list( dfr.columns )
    dropFeatrs = [ ]
    cleanNotes = { }
    
    for pos in range( 1, len( colList ) ):
        colNam = colList[ pos ]
        colSeg = dfr.iloc[ :, pos ].tolist()[ :10 ]
        remndr = set( remDict[ colNam ] )
        rMainPrint = ""
        for r in list( remndr )[ :25 ]:
            if type( r ) == float: rMainPrint = rMainPrint + f"{r}\n"
            else: rMainPrint = rMainPrint + f"{r[ :60 ]}\n"
        
        report = (f"COL [ {pos} ] {colNam}\n\n"
                  f"CLEANVALS:\n{colSeg}\n\n"
                  f"REMAINDER (unq in col: {len( remndr )}):\n{rMainPrint}\n")
        
        report_a = report + "\nACCEPT(A), BREAK(B), SCALE NOTE(C), DROP(D)"
        report_b = report_a + "\n\nPLEASE MAKE A SELECTION:\n\n"
        usinp = input( report_a )
        while usinp not in [ 'a', 'd', 'c', 'b' ]: usinp = input( report_b )
        if usinp == 'b': break
        elif usinp == 'a': continue
        elif usinp == 'd': dropFeatrs.append( colNam )
        else: cleanNotes.update( {
            colNam: input( f"{report[ :250 ]}...\n\n\nCLEANING/SCALE NOTE" ) } )
    
    def fName(ob): return f'{ob=}'.split('=')[0]
    for obj in [ dropFeatrs, cleanNotes ]:
        fTools.storePKL( obj, f'{fName(obj)}_{fTools.dtStamp()}', os.getcwd() )


def loadSDdata():
    pklFiles = [ fi for fi in
        [ open( pth, 'rb' ) for pth in
            [ list( dKey )[ 0 ] for dKey in
                [ fTools.datesortFiles( os.getcwd(), fNam ) for fNam in
                    [ 'dropFeatrs', 'cleanNotes' ] ] ] ] ]
    
    dFeats, sNotes = [ pickle.load( fi ) for fi in pklFiles ]
    for fi in pklFiles: fi.close()
    
    return dFeats, sNotes


def omitDropped(_df, dropFeats):
    # apply drop to flagged features
    dfDropped = _df.copy()
    for i in dropFeats:
        try: dfDropped.drop( [ i ], axis=1, inplace=True )
        except KeyError: pass
    return dfDropped


def flattenScale(_df, dfDct, scaleNotes, dffBook, dropFeats):
    scaleDict = {
        "million": 1000000, "billion": 1000000000, "trillion": 1000000000000 }
    
    dct = dfDct.copy()
    cleanCtry = list( _df[ 'Country' ] )
    scaleKeys = [ dkey for dkey in scaleNotes if dkey not in dropFeats ]
    
    for colName in scaleKeys:
        colVals = [ ]
        row = 0
        
        # checking remnantcol (HAS PRE-CLEAN ENTRIES) for match
        for remnt in dct[ colName ][ 'remainder' ]:
            country = dffBook[ 'Country' ][ row ]
            if country not in cleanCtry:
                row += 1
                continue
            val = _df.loc[ _df[ 'Country' ] == country ][ colName ].iloc[ 0 ]
            if type( remnt ) == float:
                colVals.append( val )
                row += 1
                continue
            if remnt.startswith( "-$" ): val = 0 - val
            
            matches = [ ]
            for scale in scaleDict:  # apply lowest-index matched scale
                try: matches.append( [ remnt.index( scale ), scale ] )
                except ValueError: continue
            if len( matches ) > 0:  # sort by lowest index (first val of match)
                matchScale = sorted( matches, key=lambda x: x[ 0 ] )[ 0 ][ 1 ]
                val = val * scaleDict[ matchScale ]
            
            colVals.append( val )
            row += 1
        
        _df[ colName ] = colVals
        return _df, cleanCtry


def showTopTen( featName, _df, asc = False, subtitle = None, unit = None ):
    print( featName )
    
    df10 = pd.concat( [ _df[ 'Country' ],
        pd.Series( _df[ featName ] ) ], axis=1 ).sort_values( by=[ featName ],
        ascending=asc )[ :10 ]
    
    fig = plt.figure( facecolor="silver" )
    ax = fig.add_axes( [ 0, 0, 1.2, 1.2 ] )
    ax.bar( df10.iloc[ :, 0 ], df10.iloc[ :, 1 ] )
    
    title = f"{'BOTTOM' if asc else 'TOP'} TEN\n{featName}"
    if subtitle: title = f"{title}\n({subtitle})"
    if unit: title = f"{title}\n[{unit}]"
    
    ax.set_title( title, fontsize=16, ha="right", weight="demi", x=0.98,
        color="black" )
    
    ax.ticklabel_format( axis='y', useOffset=False, style='plain' )
    for tick in ax.yaxis.get_major_ticks():
        tick.label.set_fontsize( 14 )
        tick.label.set_color( 'black' )
    for tick in ax.xaxis.get_major_ticks():
        tick.label.set_fontsize( 14 )
        tick.label.set_color( 'black' )
    
    plt.xticks( rotation=45, ha='right' )
    
    plt.show()





# END_INCLUDE

#