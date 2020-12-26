There are five python files listed in this folder,which are used to extract and chracterize semantic triplets from pre-processed Holocaust and Genocide testimonies from USC Shoah Foundation, Yale Fortunoff Archive, and David Boder Interviews. 

Currently, the relations among the files are:
prepocessed data --> tri_chunk_extract.py => chunk_tri_df 
      (optional) --> add_aeo.py => chunk_tri_df_wtaeo
      (optional) --> add_pvo.py => chunk_tri_df_wtaeo_wtpvo
      (optional) --> clean_add_meta.py => final_df
                 --> wrangle_final_df.py => final_df_wrangled
