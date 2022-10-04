# os ops
import osOps  # see github.com/romstroller/FileTools

# data explore / manip
from difflib import SequenceMatcher
from collections import Counter
from scipy.stats import zscore
import pandas as pd
import numpy as np
import scipy
import math
import re

# visualization / notebook graphics
from IPython.display import display, HTML
import matplotlib.pyplot as plt
import seaborn as sns


class nbData:
    """Notebook data instance. dUtls can be maintained separately"""
    
    def __init__( self ):
        self.OR = None  # raw data as originally loaded
        self.CL = None  # dataframe through cleaning phase
        self.NU = None  # dataframe for fully numeric operations
        self.DF = None  # cleaned DF for standard operations
        self.prsDct = None  # parsed column data eg. for number match
        self.ctrDct = None  # bidrectional dict for country code convert
        self.corDct = None  # correlation lookup table between all features
        self.untDct = { }  # feature unit lookup


class dUtls:
    
    def __init__( self, data ):
        self.dat = data
        self.osKit = osOps.OsKit()
        
        # pandas settings
        pd.set_option( 'display.float_format', lambda x: '%.3f' % x )
    
    def getKaggleSet( self, owner, setName ):
        self.dat.OR = self.osKit.getKaggleSet( owner, setName )
        # return self.dat.OR
    
    def tPrint( self, msg ): print( f"- [{self.osKit.dtStamp()}] {msg}" )
    
    def typeCount( self, df = None ):
        if sum( df.any() ) < 1:
            self.tPrint( "DF empty or not passed. Showing for loaded set" )
            df = self.dat.OR
        count = Counter( [ i[ 1 ] for i in df.dtypes.items() ] ).most_common()
        self.tPrint( f"Data type-count: {count}" )
    
    def generateMatchDct( self ):
        # see gist.github.com/romstroller/39dcdb3801536354e39824ac8f8f7f57
        patt = re.compile( r'([+-]?\d{1,3}(?:,\d{3})*(?:\.\d+)?|\d*\.\d+|\d+)' )
        
        dDict = { }  # collect column data
        colDex = 1
        for colName in self.dat.OR.columns[ colDex: ]:
            origCol = self.dat.OR.iloc[ :, colDex ]
            colType = self.dat.OR[ colName ].dtype
            
            # get match & unit if string, store if float
            if colType == float: colDict = {
                'matchedNums': origCol, 'remainder': [ ] }
            elif colType == np.float64: colDict = {
                'matchedNums': origCol.astype( float ), 'remainder': [ ] }
            else:
                matches, remainder = self.getMatchRemain( colDex, patt )
                colDict = { 'matchedNums': matches, 'remainder': remainder }
            
            colDict[ 'origCol' ] = origCol
            dDict[ colName ] = colDict
            colDex += 1
        
        self.tPrint( f'Got number-matches for {len( dDict.keys() )} features' )
        self.dat.prsDct = dDict
    
    def getMatchRemain( self, coIdex, patrn ):
        def excludeParenth( _mtchLi, _val ):
            mtchLi = [ ]
            # # Drop parenth'd vals (if pre-match open count > close count)
            for match in _mtchLi:
                matchDex = _val.index( match )
                openCount = _val[ :matchDex ].count( '(' )
                closCount = _val[ :matchDex ].count( ')' )
                parenthesised = openCount > closCount
                if not parenthesised: mtchLi.append( match )
            
            return mtchLi
        
        # take number-pattern match and save remainder for unit and scale
        mtches = self.dat.OR.iloc[ :, coIdex ].str.findall( patrn )
        
        pos, rmnder, mtches_ret = 0, [ ], [ ]
        for roVal in self.dat.OR.iloc[ :, coIdex ]:
            mchLi = mtches[ pos ]
            if type( mchLi ) != list: rmnt = None  # is float; no remain
            elif len( mchLi ) == 1: rmnt = roVal
            # if more than one, truncate remainder before second match
            elif len( mchLi ) > 1: rmnt = roVal[ :roVal.index( mchLi[ 1 ] ) ]
            else: rmnt = None
            
            if rmnt:
                # exclude any parenthesised matches
                mchLi = excludeParenth( mtches[ pos ], roVal )
                mtches_ret.append( mchLi )
                if len( mchLi ) > 0:
                    rmnder.append( rmnt.replace( mchLi[ 0 ], '' ) )
            else:
                if type( mchLi ) == float: mtches_ret.append( mchLi )
                else: mtches_ret.append( np.nan )
                rmnder.append( "" )
            
            pos += 1
        
        return mtches_ret, rmnder
    
    def isolateClean( self ):
        # apply split-sort to match records
        for col, colDict in self.dat.prsDct.items():
            frstVals, splitVals, checkVals = [ ], [ ], [ ]
            for mNum in range( len( colDict[ 'matchedNums' ] ) ):
                el = colDict[ 'matchedNums' ][ mNum ]
                isFilldList = (type( el ) == list) and (len( el ) > 0)
                if isFilldList:  # remove any thouscomma to support convert
                    frstVals.append( float( ''.join( el[ 0 ].split( ',' ) ) ) )
                    splitVals.append( [ v for v in el[ 1: ] ] )
                elif type( el ) == np.float64:
                    frstVals.append( float( el ) )
                else:  # check all else are either nan or empty matchlist
                    if ((type( el ) == list and len( el ) > 0) or
                        (type( el ) != list and math.isnan( el ) is False)):
                        checkVals.append( el )
                    frstVals.append( np.nan )
                    splitVals.append( np.nan )
            
            (colDict[ 'clean' ], colDict[ 'splitVals' ],
            colDict[ 'checkVals' ]) = frstVals, splitVals, checkVals
            
            # Report uncategorized data (eg val for each different type)
            if len( colDict[ 'checkVals' ] ) > 0:
                print( f"Got checkvals for {col}:" )
                typeSet = set( type( v ) for v in colDict[ 'checkVals' ] )
                egTVals = [ [ v, t ] for v, t in zip(
                    typeSet, colDict[ 'checkVals' ] ) ]
                for t, v in egTVals: print( f"[{v}] is [{t}]\n" )
        
        self.tPrint( "Isolated matched floats" )
    
    def getCleanDF( self ):
        # dictionary columns to DF, checking is now float
        newCols = [ ]
        cleanDF = self.dat.OR.iloc[ :, 0 ]  # start with countries
        for col in self.dat.prsDct:
            clean = pd.Series( self.dat.prsDct[ col ][ 'clean' ] )
            lenFloat = len( [ i for i in clean if type( i ) == float ] )
            if lenFloat > len( clean ) * 0.90:
                newCols.append( col )
                cleanDF = pd.concat( [ cleanDF, clean ], axis=1 )
            else: print( "col is less than 90% float. Dropping..." )
        
        cleanDF.columns = [ 'Country' ] + newCols
        
        self.dat.CL = cleanDF
        self.tPrint( "Got dataframe from float data" )
    
    def getNumericNonNan( self ):
        # Enforce non-nan threshold for dimensions ( av. dense + .5 st.dev )
        #   convert to numeric, add featname row to track through clean
        self.dat.CL.loc[ -1 ] = self.dat.CL.columns
        self.dat.CL.index = self.dat.CL.index + 1
        self.dat.CL.sort_index( inplace=True )
        
        nonNans = [ ]
        for pos in range( 0, self.dat.CL.shape[ 1 ] ):
            vals = self.dat.CL.iloc[ :, pos ].tolist()
            nonNans.append( [ vals, len( [ v for v in vals
                if type( v ) == float and not math.isnan( v ) ] ) ] )
        
        nLi = [ nval for _, nval in nonNans ]
        # non-nan threshold of average plus .5 standard deviation (rounded)
        _thresh = int( (sum( nLi ) / len( nLi )) + 0.5 * np.std( nLi ) )
        keepCols = [ kval for kval, nnul in nonNans if nnul >= _thresh ]
        dfNum = pd.DataFrame( { col[ 0 ]: col[ 1: ] for col in keepCols } )
        dfNum = dfNum.apply( pd.to_numeric, errors='ignore' )
        dfNum.insert( 0, 'Country', self.dat.CL.iloc[ :, 0 ].tolist()[ 1: ] )
        self.dat.CL = dfNum
        self.tPrint( "Enforced non-NaN threshold in data" )
    
    def cleanReport( self, dfLi = None ):
        if not dfLi: dfPre, dfPost = self.dat.OR, self.dat.CL
        else:  dfPre, dfPost = dfLi
        
        preIsNa = dfPre.isna().sum().sum()
        posIsNa = dfPost.isna().sum().sum()
        preDim = dfPre.shape[ 0 ] * dfPre.shape[ 1 ]
        posDim = dfPost.shape[ 0 ] * dfPost.shape[ 1 ]
        print(
            f"    BEFORE shape: {dfPre.shape}, "
            f"NAN-density: {(preIsNa / preDim) * 100:.2f}% "
            f"({preIsNa} NaN in {preDim} values)\n"
            f"    AFTER shape: {dfPost.shape}, "
            f"NAN-density: {(posIsNa / posDim) * 100:.2f}% "
            f"({posIsNa} NaN in {posDim} values)\n" )
    
    def runScaleAnalysis( self, dfr, remDict ):
        colList = list( dfr.columns )
        dropFeatrs, cleanNotes = [ ], { }
        
        for pos in range( 1, len( colList ) ):
            colNam = colList[ pos ]
            colSeg = dfr.iloc[ :, pos ].tolist()[ :10 ]
            remndr = set( remDict[ colNam ] )
            rMainPrint = ""
            for r in list( remndr )[ :25 ]:
                if type( r ) == float: rMainPrint = rMainPrint + f"{r}\n"
                else: rMainPrint = rMainPrint + f"{r[ :60 ]}\n"
            
            report = (f"COL [ {pos} ] {colNam}\n\n"
                      f"CLNVALS:\n{colSeg}\n\n"
                      f"REMNDER: {len( remndr )}):\n{rMainPrint}\n")
            
            report_a = report + "\nACCEPT(A), BREAK(B), SCALE NOTE(C), DROP(D)"
            report_b = report_a + "\n\nPLEASE MAKE A SELECTION:\n\n"
            usinp = input( report_a )
            while usinp not in [ 'a', 'd', 'c', 'b' ]: usinp = input( report_b )
            if usinp == 'b': break
            elif usinp == 'a': continue
            elif usinp == 'd': dropFeatrs.append( colNam )
            else: cleanNotes.update( {
                colNam: input( f"{report[ :250 ]}\n\n\nCLEAN/SCALE NOTE" ) } )
        
        def fName( ob ): return f'{ob=}'.split( '=' )[ 0 ]
        
        for obj in [ dropFeatrs, cleanNotes ]:
            self.osKit.storePKL( obj, f'{fName( obj )}_{self.osKit.dtStamp()}' )
    
    def unPklData( self, *args, **kwargs ):
        self.tPrint( f"Returned {[ a for a in args ]} from PKL" )
        return self.osKit.unPklData( *args, **kwargs )
    
    def runDrops( self, dropFeats ):
        # apply drop to flagged features
        dfDropped = self.dat.CL.copy()
        for i in dropFeats:
            try: dfDropped.drop( [ i ], axis=1, inplace=True )
            except KeyError: pass
        self.dat.CL = dfDropped
        self.tPrint( "Dropped features as marked" )
    
    def flattenScale( self, scaleNotes, dropFeats ):
        scaleDict = {
            "million": 1000000,
            "billion": 1000000000,
            "trillion": 1000000000000 }
        
        cleanCtry = list( self.dat.CL[ 'Country' ] )
        scaleKeys = [ dkey for dkey in scaleNotes if dkey not in dropFeats ]
        
        for colName in scaleKeys:
            colVals, row = [ ], 0
            
            # checking remnantcol (HAS PRE-CLEAN ENTRIES) for match
            for remnt in self.dat.prsDct[ colName ][ 'remainder' ]:
                country = self.dat.OR[ 'Country' ][ row ]
                row += 1
                
                if country not in cleanCtry: continue
                val = self.dat.CL.loc[
                    self.dat.CL[ 'Country' ] == country ][ colName ].iloc[ 0 ]
                if type( remnt ) == float:
                    colVals.append( val )
                    continue
                if remnt.startswith( "-$" ): val = 0 - val
                
                matches = [ ]
                for scale in scaleDict:  # apply lowest-index matched scale
                    try: matches.append( [ remnt.index( scale ), scale ] )
                    except ValueError: continue
                if len( matches ) > 0:  # sort by low-index (first match val)
                    mScale = sorted( matches, key=lambda x: x[ 0 ] )[ 0 ][ 1 ]
                    val = val * scaleDict[ mScale ]
                
                colVals.append( val )
            
            self.dat.CL[ colName ] = colVals
        
        self.tPrint( "Flattened scale-variant values" )
    
    def popRowsByFtVal( self, ftNam, vals ):
        df_OrCols = self.dat.CL.columns
        dexVals = self.dat.CL.iloc[ :, 0 ].values.tolist()  # 1st colVals as i
        popDexs = [ self.dat.CL.index[ self.dat.CL[ ftNam ] == val ].tolist()
            for val in vals ]
        df_t = self.dat.CL.T
        df_t.columns = self.dat.CL[ df_OrCols[ 0 ] ].tolist()
        
        for group in popDexs:
            for val in group: df_t.pop( dexVals[ val ] )
        
        self.dat.CL = df_t.drop( df_t.columns[ 0 ], axis=1 ).T
    
    def numercisedDF( self ):
        self.dat.ctrDct = biDict( { i: c for i, c in enumerate(
            self.dat.CL[ 'Country' ].tolist() ) } )
        self.dat.CL[ 'Country' ].replace( self.dat.ctrDct.i, inplace=True )
        self.dat.NU = self.dat.CL.apply( pd.to_numeric )  # for stat funcs
        self.dat.DF = self.dat.NU.copy()  # for maxima-minima analysis
        self.dat.DF[ 'Country' ].replace( self.dat.ctrDct, inplace=True )
        self.tPrint( 'Numericised features' )
    
    def generateUnitDct( self ):
        """From feats in (limited) numericised frame, unit segs from OR"""
        
        (fts := list( self.dat.NU.columns )).remove( 'Country' )
        
        print( "Getting unit dct" )
        
        untDct = { }
        colDex = 0
        
        for ft in fts[ colDex: ]:
            print( f"getting units for:\n{ft}" )
            colType = self.dat.OR[ ft ].dtype
            if colType in (float, np.float64): untDct.update( { ft: None } )
            else: untDct.update( { ft: self.matchUnits( ft ) } )
            colDex += 1
        
        print( " COMPLETED get unit dct" )
        
        self.osKit.storePKL( untDct, 'unitDct', stamp=True )
    
    def matchUnits( self, _ft ):
        """returns (by count) recurring substrings in df column"""
        
        def matchSeg( s1, s2 ):
            return (SequenceMatcher( None, s1, s2 ).find_longest_match()
                    if (float not in [ type( s1 ), type( s2 ) ]) else None)
        
        units = [ ]
        for xRow in range( 1, self.dat.OR.shape[ 0 ] ):
            for yRow in range( xRow + 1, self.dat.OR.shape[ 0 ] ):
                if m := matchSeg( sX := self.dat.OR.loc[ xRow, _ft ],
                    sY := self.dat.OR.loc[ yRow, _ft ] ):
                    units += [ sX[ m.b:m.b + m.size ], sY[ m.b:m.b + m.size ] ]
        
        if len( c := list( Counter( units ).most_common() ) ) > 0: return c
        else: return None
    
    def cleanUnits( self, _unts, ):
        dfO = self.dat.OR
        
        def checkMatchHalf( col, pat ):
            mSize = dfO[ dfO[ col ].str.contains( pat, na=False ) ].shape[ 0 ]
            return mSize > (dfO.shape[ 0 ] / 2)
        
        unitPatDct = {
            '(mal': { 'patt': re.compile( r'%.*\(male[^/)]*/female[^)]*\)' ),
                'unit': "% (m/f)" },
            'male(': { 'patt': re.compile( r'male\(s\)/female' ),
                'unit': "m/f" },
            'migrant': { 'patt': re.compile( r'migrant\(s\)/1,000 p' ),
                'unit': "migrants/1,000 people" } }
        
        for k, v in _unts.items():
            unt = None
            if not v: continue  # no common string (col was float)
            for key, patD in unitPatDct.items():  # unit from re patt if segmt
                if True in [ key in vv[ 0 ] for vv in v[ :5 ] ]:
                    # checkMatchHalf( k, patD[ 'patt' ] )
                    if checkMatchHalf( k, patD[ 'patt' ] ):
                        unt = patD[ 'unit' ]
                if unt: break
            if unt: self.dat.untDct.update( { k: unt } )
            else:
                unt = v[ 0 ][ 0 ]
                # end at brackopen, end after "pop", remove lead num & space
                unt = unt.split( '(' )[ 0 ]
                if (p := 'population') in unt: unt = unt.split( p )[ 0 ] + p
                while unt and ((u := unt[ 0 ]).isdigit() or (u == ',')):
                    unt = unt[ 1: ]
                self.dat.untDct.update( { k: unt.strip() } )
        
        self.tPrint( "Supplied cleaned unit strings" )
    
    def showPDens( self, _ft ):
        print( f"PROBABILITY DENSITY FOR:\n{_ft}" )
        
        def map_pdf( _x, **kwargs ):
            if not kwargs: pass
            mu, std = scipy.stats.norm.fit( _x )
            x0, x1 = p1.axes[ 0 ][ 0 ].get_xlim()
            x_pdf = np.linspace( x0, x1, 100 )
            y_pdf = scipy.stats.norm.pdf( x_pdf, mu, std )
            plt.plot( x_pdf, y_pdf, c='r' )
            plt.xlabel( f"{_ft} ({self.dat.untDct[ _ft ]})" )
            plt.show()
        
        _df = pd.DataFrame( { _ft: [ v for v in
            self.dat.NU[ _ft ].values.tolist() if np.isfinite( v ) ] } )
        
        p1 = sns.displot(
            data=_df, x=_ft, kind='hist', bins=_df.shape[ 0 ], stat='density')
        p1.figure.set_figwidth( 16 )
        p1.figure.set_figheight( 9 )
        p1.figure.set_facecolor( 'Silver' )
        p1.map( map_pdf, _ft )
    
    def getZThreshDF( self, _ft, zT, excl = False, asc = False, ret = False ):
        # get country and value for non-nan values of selected feature
        vLi = [ [ c, v ] for c, v in zip(
            self.dat.NU[ 'Country' ].values.tolist(),
            self.dat.NU[ _ft ].values.tolist() ) if np.isfinite( v ) ]
        
        # get ctry int, name, value and z-score for values
        dfCi = pd.DataFrame( c for c, n in vLi )
        dfCs = pd.DataFrame( self.dat.ctrDct[ c ] for c, n in vLi )
        dfZ = (dfV := pd.DataFrame( n for c, n in vLi )).apply( zscore )
        
        # get original string value from raw-loaded dataset
        dfO = pd.DataFrame( self.dat.OR.loc[
            self.dat.OR[ 'Country' ] == self.dat.ctrDct[ c ], _ft ].iloc[ 0 ]
            for c, n in vLi )
        
        # concat as df and limit to outside +- zScore threshold
        dfCZ = pd.concat( [ dfCi, dfCs, dfZ, dfV, dfO ], axis=1 )
        dfCZ.columns = [ 'ctry_i', 'ctry_s', 'zScore', 'value', 'orVal' ]
        
        # if excl ("exclude"), filter df within threshold instead out outliers
        dfL = (dfCZ.loc[ ((dfCZ.zScore >= zT) | (dfCZ.zScore <= -zT)) ]
               if not excl else
               dfCZ.loc[ ((dfCZ.zScore <= zT) & (dfCZ.zScore >= -zT)) ])
        
        dfSort = dfL.sort_values( 'zScore', ascending=asc )
        
        if ret: return dfSort
        else:
            inc = 'OUTSIDE' if not excl else 'INSIDE'
            print( f"Z_SCORES {inc} >+/<-[ {zT} ] for non-NaNs in:\n{_ft}" )
            display( dfSort )
    
    def getCorDct( self ):
        """ Generate CORRELATION DICTIONARY where keys are feature pairs,
            (pairs as frozensets for reversible feat lookup)
            values are correlations. """
        fset = frozenset
        corDict = { }
        
        bCol = 1
        df = self.dat.DF
        while bCol < df.shape[ 1 ]:
            # for each feat, get any correls for each feat to the right
            for iCol in range( bCol + 1, self.dat.DF.shape[ 1 ] ):
                corrs = df.iloc[ :, bCol ].corr( df.iloc[ :, iCol ] )
                corDict.update( { fset( [ bCol, iCol ] ): corrs } )
            
            bCol += 1
            # if bCol == df.shape[ 1 ]:
            #     self.tPrint( f"Compiled {len( corDict )} correlations" )
            #     self.dat.corDct = corDict
        self.tPrint( f"Compiled {len( corDict )} correlations" )
        self.dat.corDct = biDict( { k: v for k, v in corDict.items() } )
    
    def getCTDct( self, inLim, outLim = float( 'inf' ) ):
        # Examine features with correlations within specified threshold
        # explore different thesholds
        """collect feature correlations within specified significance range"""
        threshDict = { }
        for key, cor in self.dat.corDct.items():
            bCol = list( key )[ 0 ]
            cCol = list( key )[ 1 ]
            if (-outLim < cor <= -inLim) or (inLim <= cor < outLim):
                threshDict[ key ] = {
                    'corr': cor,
                    'inn_lim': inLim,
                    'baseCol': bCol,
                    'compCol': cCol,
                    'out_lim': outLim if outLim != float( 'inf' ) else "inf",
                    'baseName': self.dat.DF.columns[ bCol ],
                    'compName': self.dat.DF.columns[ cCol ] }
        return threshDict
    
    def dropDupCorrs( self ):
        # Check perfect correlations. Confirmed duplicates, drop
        self.tPrint( 'Dropping duplicate features' )
        for i in (cDct := self.getCTDct( 1 )):
            corr = cDct[ i ][ 'corr' ]
            ftrX, ftrY = cDct[ i ][ 'baseName' ], cDct[ i ][ 'compName' ]
            print( f"    {corr=}\n    {ftrX=}\n    {ftrY=}\n" )
            self.dat.corDct.pop( i )
    
    def showMaxima( self, ft, n = 10, asc = False, sub = None, unit = None,
        df = None, mask = None, short = False ):
        # check non-default DF or df mask
        if not df: df = self.dat.DF
        try: df = df.loc[ mask ] if mask.any() else df
        except AttributeError: pass
        
        df10 = pd.concat( [ df[ 'Country' ],
            pd.Series( df[ ft ] ) ], axis=1 ).sort_values( by=[ ft ],
            ascending=asc )[ :n ]
        
        fig = plt.figure( facecolor="silver" )
        height = 3.0 if short else 1.6
        ax = fig.add_axes( [ 0, 0, height, 1.2 ] )
        bars = ax.bar( df10.iloc[ :, 0 ], df10.iloc[ :, 1 ], edgecolor="black" )
        
        n = n if (le := df.shape[ 0 ]) >= n else le  # if small df
        title = f"{'BOTTOM' if asc else 'TOP'} {n}\n{ft}"  # adjust label
        if sub: title = f"{title}\n({sub})"
        
        # unit: default add from dict
        if not unit:
            try: unit = self.dat.untDct[ ft ]
            except KeyError: unit = "\n\nUNIT NOT SET"
        if unit == "_": unit = None
        
        title = f"{title}\n{unit}" if unit else title
        
        ax.set_title( title, fontsize=16, ha="right", weight="demi", x=0.98,
            color="black" )
        
        ax.ticklabel_format( axis='y', useOffset=False, style='plain' )
        for tick in ax.yaxis.get_major_ticks() + ax.xaxis.get_major_ticks():
            tick.label.set_fontsize( 14 )
            tick.label.set_color( 'black' )
        
        plt.xticks( rotation=45, ha='right' )
        
        def gradientbars_sliced( _bars ):
            _ax = _bars[ 0 ].axes
            xmin, xmax = _ax.get_xlim()
            ymin, ymax = _ax.get_ylim()
            for bar in _bars:
                bar.set_zorder( 1 )
                bar.set_facecolor( "none" )
                x, y = bar.get_xy()
                w, h = bar.get_width(), bar.get_height()
                grad = np.linspace( y, y + h, 256 ).reshape( 256, 1 )
                _ax.imshow( grad, extent=[ x, x + w, y, y + h ], aspect="auto",
                    zorder=0, origin='lower', vmin=ymin, vmax=ymax,
                    cmap='magma' )
            _ax.axis( [ xmin, xmax, ymin, ymax ] )
        
        gradientbars_sliced( bars )
        
        plt.show()
    
    def plotScttr( self, fts, fts2 = None, df = None, mask = None, uns = None ):
        if not df: df = self.dat.DF
        try: df = df.loc[ mask ] if mask.any() else df
        except AttributeError: pass
        
        # unit default from dict, fail "NOT SET", uscore to skip
        if not uns:
            try: uns = [ self.dat.untDct[ ft ] for ft in fts ]
            except KeyError: uns = [ "\n\n"+(na := "UNIT NOT SET"), na ]
        if uns == "_": uns = [ None, None ]
        
        xTit, yTit = [ f"{ft} ({u})" for ft, u in zip(fts, uns) ]
        
        if fts2:
            fig, (ax1, ax2) = plt.subplots( 1, 2 )  # sharey='all'
            ax1.scatter( df[ fts[ 0 ] ], df[ fts[ 1 ] ], c='black' )
            ax2.scatter( df[ fts2[ 0 ] ], df[ fts2[ 1 ] ], c='black' )
            for side, g in zip( [ 'LEFT', 'RIGHT' ], [ fts, fts2 ] ):
                for f, ax in zip( g, [ 'X', 'Y' ] ):
                    print( f"{side} {ax}: {f}" )
        else:
            x, y = df[ fts[ 0 ] ], df[ fts[ 1 ] ]
            plt.figure( figsize=(16, 9), facecolor="silver" )
            plt.plot( x, y, 'o', color='black' )
            plt.xlabel( xTit )
            plt.ylabel( yTit )
            print( f"Feats: [ {fts[ 0 ]} ]\n       [ {fts[ 1 ]} ]" )
        
        plt.show()
    
    def getRank( self, ctry, ft, asc = False ):
        df = self.dat.DF
        value = df[ ft ].loc[ df[ 'Country' ] == ctry ].values[ 0 ]
        if str( value ) == 'nan': return print( f"{ctry} is null for\n{ft}" )
        
        def ranks( val, v ): return (val < v) if not asc else (val > v)
        
        order = "DESCENDING" if not asc else "ASCENDING"
        
        rank = len( [ v for v in pd.Series( df[ ft ] ) if ranks( value, v ) ] )
        ties = len( [ v for v in pd.Series( df[ ft ] ) if v == value ] ) - 1
        
        print( f"With value of [ {value} ], {ctry} is ranked {rank} for:\n"
               f"    '{ft}'\n    (total {df.shape[ 0 ]}, ranked {order})" )
        if ties > 0: print( f"    TIED WITH {ties} COUNTRIES" )
    
    def dfPrint( self, _df = None ):
        if not _df: _df = self.dat.DF
        # # options at https://pandas.pydata.org/docs/user_guide/options.html
        with pd.option_context(
            'notebook_repr_html', True, ):
            display( _df )
            # print( _df )
    
    def sDevOutliers( self, _ft, sdThresh = 3 ):
        """ simple s-Dev outlier identification, superseded by getZThreshDF
            eg: ftOut, ftFilt = dataUtls.sDevOutliers(
                'Geography: Area - total', sdThresh=2.5 ) """
        f64 = np.float64
        data = np.asarray( self.dat.DF[ _ft ].dropna() )
        d_mean, d_sDev = np.mean( data, dtype=f64 ), np.std( data, dtype=f64 )
        lower, upper = d_mean - (lim := d_sDev * sdThresh), d_mean + lim
        outs = [ x for x in data if x < lower or x > upper ]
        filt = [ x for x in data if lower <= x <= upper ]
        print( f"ftLen={len( data )}\n{d_mean=}\n{d_sDev=}\n{lim=}\n{lower=}\n"
               f"{upper=}\nlen(outs)={len( outs )}\nlen(filt)={len( filt )}" )
        return sorted( outs, reverse=True ), filt
    
    def getVal( self, _ctry, _feat ):
        return self.dat.DF.loc[
            self.dat.DF[ 'Country' ] == _ctry, _feat ].iloc[ 0 ]
    
    def searchFeatures( self, seg ):
        return [ c for c in self.dat.DF.columns if seg in c ]
    
    def frozSetFromFeats( self, f1, f2 ):
        return frozenset(
            [ list( self.dat.DF.columns ).index( c ) for c in [ f1, f2 ] ] )
    
    def reportCorr( self, cfts ):
        bCol, cPos = [ list( self.dat.DF.columns ).index( ft ) for ft in cfts ]
        return f"LIN. CORR: {self.dat.corDct[ frozenset( [ bCol, cPos ] ) ]}"
    
    def cycleT10( self, _df, start = 0, showN = 1, asc = False ):
        # iterate feats through T10 analysis (progress)
        print( f"FEAT {start}-{start + showN} of {len( _df.columns )}" )
        for i in list( _df.columns )[ start:start + showN ]:
            self.showMaxima( i, _df, asc )
            start += showN
    
    def reportDiffs( self, tDict ):
        """Identify and report value differences between correlated features"""
        
        def report( tDct, _key ):
            """output feature detail for sig range """
            item = tDct[ _key ]
            return (
                f"\nCORRELATION FOR FEAT-PAIR {str( list( _key ) )}"
                f"\nIN THRESH +-=[ {item[ 'inn_lim' ]}-{item[ 'out_lim' ]} ] "
                f"\nCORR: {item[ 'corr' ]}"
                f"\nBASE: {item[ 'baseName' ]}"
                f"\nCOMP: {item[ 'compName' ]}")
        
        difDct = { }
        for k in tDict:
            bCol, cCol = [ self.dat.DF.columns[ f ] for f in list( k ) ]
            dfCompare = self.dat.DF[ [ bCol, cCol ] ].loc[
                ~(self.dat.DF[ bCol ] == self.dat.DF[ cCol ]) ]
            if len( dfCompare.dropna() ) < 1:
                print( f"{report( tDict, k )}" )
                print( f"{list( k )}: DIFFERENCES ALL NaN\n" )
            else: difDct.update( { k: dfCompare } )
        if len( difDct ) < 1: print( "\nNON-NAN DIFFS FOR FEATS" )
        for k in difDct:
            difDf = difDct[ k ].dropna()
            print( f"{report( tDict, k )}" )
            print( f"{len( difDf )} Non-NaN diffs for {list( k )}. First 3:" )
            display( HTML( difDf[ :3 ].to_html() ) )
    
    def reportStrongNeg( self, sigThresh ):
        nDatFts = list( self.dat.DF.columns )
        sortdCorrs = sorted( [ k for k in list( self.dat.corDct.i.keys() ) ] )
        
        # Exclude birthrate/urbaniztn version
        collect, pos, exc = { }, 0, 0
        excluded = [ ': Birth rate', ': Urbanization' ]
        inThresh = True
        while inThresh:
            k = sortdCorrs[ pos ]
            ftFSetLi = self.dat.corDct.i[ k ]
            for fSetX, fSetY in ftFSetLi:
                if True in [ x in y for x in excluded for y in [
                    nDatFts[ fSetX ], nDatFts[ fSetY ] ] ]:
                    exc += 1
                    continue
                else: collect.update(
                    { k: [ nDatFts[ fSetX ], nDatFts[ fSetY ] ] } )
            if k > sigThresh: inThresh = False
            pos += 1
        
        n = "All" if ((L := len( collect )) == 0) else L
        print( f"{n} of {exc} negative correlations stronger than {sigThresh=} "
               f"\n    were for feature pairs that included at least one of "
               f"\n    {excluded=}" )
    
    def setMean( self, _ft, ctry, dfs ):
        # Set feature value for Country to mean ( feature excl. Ctry )
        
        nonCtryMean = self.dat.DF.loc[
            self.dat.DF[ 'Country' ] != ctry ][ _ft ].mean()
        
        for i, df in enumerate( dfs ):
            ctry = ctry if i == 0 else self.dat.ctrDct.i[ ctry ][ 0 ]
            df.iat[
                (df.Country.to_list()).index( ctry ),
                (df.columns.to_list()).index( _ft ) ] = nonCtryMean


class biDict( dict ):
    """ modified from author stackoverflow.com/users/1422096/basj
        access inverse with myDict.i['key']
    """
    
    def __init__( self, *args, **kwargs ):
        super( biDict, self ).__init__( *args, **kwargs )
        self.inverse = self.i = { }  # added shorter ref
        for key, value in self.items():
            self.inverse.setdefault( value, [ ] ).append( key )
    
    def __setitem__( self, key, value ):
        if key in self: self.inverse[ self[ key ] ].remove( key )
        super( biDict, self ).__setitem__( key, value )
        self.inverse.setdefault( value, [ ] ).append( key )
    
    def __delitem__( self, key ):
        self.inverse.setdefault( self[ key ], [ ] ).remove( key )
        if self[ key ] in self.inverse and not self.inverse[ self[ key ] ]:
            del self.inverse[ self[ key ] ]
        super( biDict, self ).__delitem__( key )

# END_INCLUDE

#
