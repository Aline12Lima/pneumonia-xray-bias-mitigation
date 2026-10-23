# PadChest — auditoria preliminar (sem imagens)

## 1. CSV
- Linhas: 160,861
- Colunas (36): Unnamed: 0, ImageID, ImageDir, StudyDate_DICOM, StudyID, PatientID, PatientBirth, PatientSex_DICOM, ViewPosition_DICOM, Projection, MethodProjection, Pediatric, Modality_DICOM, Manufacturer_DICOM, PhotometricInterpretation_DICOM, PixelRepresentation_DICOM, PixelAspectRatio_DICOM, SpatialResolution_DICOM, BitsStored_DICOM, WindowCenter_DICOM, WindowWidth_DICOM, Rows_DICOM, Columns_DICOM, XRayTubeCurrent_DICOM, Exposure_DICOM, ExposureInuAs_DICOM, ExposureTime, RelativeXRayExposure_DICOM, ReportID, Report, MethodLabel, Labels, Localizations, LabelsLocalizationsBySentence, labelCUIS, LocalizationsCUIS

## 2. Pacotes (listagens unzip -l)
- Listagens lidas: 52 pacotes; PNG listados: 160,861
- Pacotes encontrados: [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23, 24, 25, 26, 27, 28, 29, 30, 31, 32, 33, 34, 35, 36, 37, 38, 39, 40, 41, 42, 43, 44, 45, 46, 47, 48, 49, 50, 54]
- Tamanho total listado: 1,058.1 GB; média por imagem: 6.6 MB

## 3. Conferência CSV × listagens
- Imagens do CSV encontradas em alguma listagem: 160,861 (100.0%)
- Com ImageDir igual ao pacote da listagem: 160,861 (100.0% das encontradas)
- PNG listados que não estão no CSV: 0

## 4. Verify_Zips_ImageCounts (primeiras linhas)
|   Zip |   ImageCounts |
|------:|--------------:|
|     0 |          3004 |
|     1 |          3024 |
|     2 |          2984 |
|     3 |          2992 |
|     4 |          2993 |
|     5 |          2947 |
|     6 |          1645 |
|     7 |          2742 |
|     8 |          2991 |
|     9 |          2990 |
|    10 |          2975 |
|    11 |          3022 |
|    12 |          3217 |
|    13 |          3218 |
|    14 |          3217 |
|    15 |          3217 |
|    16 |          3217 |
|    17 |          3217 |
|    18 |          3218 |
|    19 |          3217 |
|    20 |          3217 |
|    21 |          3217 |
|    22 |          3218 |
|    23 |          3217 |
|    24 |          3217 |
|    25 |          3217 |
|    26 |          3217 |
|    27 |          3218 |
|    28 |          3217 |
|    29 |          3217 |
|    30 |          3217 |
|    31 |          3218 |
|    32 |          3217 |
|    33 |          3217 |
|    34 |          3217 |
|    35 |          3217 |
|    36 |          3218 |
|    37 |          3217 |
|    38 |          3217 |
|    39 |          3217 |
|    40 |          3218 |
|    41 |          3217 |
|    42 |          3217 |
|    43 |          3217 |
|    44 |          3217 |
|    45 |          3218 |
|    46 |          3217 |
|    47 |          3217 |
|    48 |          3217 |
|    49 |          3217 |
|    50 |             1 |
|    54 |          4297 |

## 5. Campos DICOM no CSV (passo 3.7) — todas as imagens
|                                 |   preenchido_% |   n_valores | exemplos                                         |
|:--------------------------------|---------------:|------------:|:-------------------------------------------------|
| StudyDate_DICOM                 |          100   |        2789 | 20140505, 20140508, 20140422, 20140428           |
| PatientSex_DICOM                |          100   |           3 | M, F, O                                          |
| ViewPosition_DICOM              |           57.1 |          10 | POSTEROANTERIOR, LATERAL, PA, LL                 |
| Modality_DICOM                  |          100   |           2 | CR, DX                                           |
| Manufacturer_DICOM              |          100   |           2 | PhilipsMedicalSystems, ImagingDynamicsCompanyLtd |
| PhotometricInterpretation_DICOM |          100   |           1 | MONOCHROME2                                      |
| PixelRepresentation_DICOM       |          100   |           1 | 0                                                |
| PixelAspectRatio_DICOM          |           29.8 |           1 | ['1', '1']                                       |
| SpatialResolution_DICOM         |           40   |           9 | 0.143, 0.2, 0.148, 0.2000000029802               |
| BitsStored_DICOM                |          100   |           2 | 12, 10                                           |
| WindowCenter_DICOM              |           99.2 |        1596 | 2047.0, 511.5, 2047.5, 2046.0                    |
| WindowWidth_DICOM               |           99.2 |        2148 | 4095.0, 1023.0, 2314.0, 2311.0                   |
| Rows_DICOM                      |          100   |        1933 | 1760, 3520, 3000, 4280                           |
| Columns_DICOM                   |          100   |        1961 | 2140, 2000, 4280, 3000                           |
| XRayTubeCurrent_DICOM           |           64   |          10 | 250.0, 0.0, 400.0, 320.0                         |
| Exposure_DICOM                  |           70.5 |          46 | 2.0, 0.0, 3.0, 12.0                              |
| ExposureInuAs_DICOM             |           56.1 |         212 | 2500.0, 3200.0, 12500.0, 10000.0                 |
| ExposureTime                    |           70.5 |         107 | 10.0, 0.0, 25.0, 5.0                             |
| RelativeXRayExposure_DICOM      |           86.7 |        2693 | -1.2, -1.16, -1.03, -0.99                        |

## 6. Projeções (todas as imagens)
| Projection    |     n |
|:--------------|------:|
| PA            | 91728 |
| L             | 49579 |
| AP_horizontal | 14346 |
| AP            |  4559 |
| COSTAL        |   630 |
| EXCLUDE       |    11 |
| UNK           |     8 |

## 7. Fluxograma de exclusões (provisório)
| etapa                                                        |      n |
|:-------------------------------------------------------------|-------:|
| Todas as imagens                                             | 160861 |
| Frontais ['PA', 'AP', 'AP_horizontal']                       | 110633 |
| Sem marca Pediatric=PED                                      | 110383 |
| Idade ≥ 18 (ano do estudo − ano de nascimento)               | 105108 |
| Com rótulo                                                   | 104585 |
| Com pacote identificado                                      | 104585 |
| Uma frontal por estudo (provisório: PA > AP > AP_horizontal) | 103036 |

## 8. Pneumonia no recorte
- Estudos: 103,036; pacientes: 62,714
- Positivos (rótulo exato 'pneumonia'): 4,079 (4.0%)
- Com qualquer termo contendo 'pneumonia': 4,167
- Rótulo 'exclude' presente: 892
- Termos com 'pneumonia':
|                    |    n |
|:-------------------|-----:|
| pneumonia          | 4123 |
| atypical pneumonia |   97 |

- Por método de rotulagem (controle do passo 7.7):
| MethodLabel   |   estudos |   positivos |   prevalencia_% |
|:--------------|----------:|------------:|----------------:|
| Physician     |     25123 |         805 |             3.2 |
| RNN_model     |     77913 |        3274 |             4.2 |

- Por projeção:
| Projection    |   estudos |   positivos |   prevalencia_% |
|:--------------|----------:|------------:|----------------:|
| PA            |     86543 |        2670 |            3.09 |
| AP_horizontal |     13350 |        1157 |            8.67 |
| AP            |      3143 |         252 |            8.02 |

- Por fabricante:
| Manufacturer_DICOM        |   estudos |   positivos |   prevalencia_% |
|:--------------------------|----------:|------------:|----------------:|
| ImagingDynamicsCompanyLtd |     53080 |         955 |            1.8  |
| PhilipsMedicalSystems     |     49956 |        3124 |            6.25 |

- Por sexo:
| PatientSex_DICOM   |   estudos |   positivos |   prevalencia_% |
|:-------------------|----------:|------------:|----------------:|
| M                  |     51624 |        2352 |            4.56 |
| F                  |     51394 |        1727 |            3.36 |
| O                  |        15 |           0 |            0    |

- Por ano do estudo:
|   study_year |   estudos |   positivos |
|-------------:|----------:|------------:|
|         2007 |        15 |           1 |
|         2008 |        27 |           1 |
|         2009 |     13936 |         682 |
|         2010 |     14618 |         708 |
|         2011 |     14830 |         723 |
|         2012 |     15054 |         541 |
|         2013 |     14448 |         491 |
|         2014 |     10996 |         311 |
|         2015 |      7050 |         249 |
|         2016 |      6857 |         210 |
|         2017 |      5205 |         162 |

## 9. Resumo por pacote (recorte)
|   zip |   estudos |   pacientes |   positivos |   prev_% |   ano_min |   ano_max |   %PA |   %medico |   idade_media |   GB |
|------:|----------:|------------:|------------:|---------:|----------:|----------:|------:|----------:|--------------:|-----:|
|     0 |      1888 |        1782 |          63 |     3.34 |      2009 |      2017 |  91.6 |     100   |          62.1 | 21.9 |
|     1 |      1945 |        1804 |          79 |     4.06 |      2009 |      2017 |  91.5 |     100   |          62   | 22   |
|     2 |      1940 |        1807 |          64 |     3.3  |      2009 |      2017 |  90.2 |     100   |          62.2 | 21.8 |
|     3 |      1930 |        1795 |          74 |     3.83 |      2009 |      2017 |  92   |     100   |          61.6 | 21.7 |
|     4 |      1917 |        1776 |          71 |     3.7  |      2009 |      2017 |  92.4 |     100   |          61.1 | 21.9 |
|     5 |      1876 |        1743 |          43 |     2.29 |      2009 |      2017 |  92.3 |     100   |          62.1 | 21.7 |
|     6 |      1054 |         985 |          37 |     3.51 |      2009 |      2017 |  92.2 |     100   |          61.8 | 12.2 |
|     7 |      1778 |        1648 |          66 |     3.71 |      2009 |      2017 |  90.7 |     100   |          61.7 | 20.1 |
|     8 |      1936 |        1809 |          59 |     3.05 |      2009 |      2017 |  92.3 |     100   |          62.1 | 21.7 |
|     9 |      1931 |        1810 |          52 |     2.69 |      2009 |      2017 |  92.2 |     100   |          62.1 | 21.7 |
|    10 |      1938 |        1811 |          57 |     2.94 |      2009 |      2017 |  91.7 |     100   |          61.3 | 21.9 |
|    11 |      1944 |        1811 |          59 |     3.03 |      2009 |      2017 |  91.6 |     100   |          61.9 | 22.2 |
|    12 |      2015 |        1871 |          70 |     3.47 |      2009 |      2017 |  83.6 |      14.4 |          62.1 | 22.5 |
|    13 |      2147 |        2015 |          81 |     3.77 |      2010 |      2013 |  82.9 |       0   |          62.2 | 20.2 |
|    14 |      2101 |        1952 |          64 |     3.05 |      2013 |      2013 |  84.6 |       0   |          62   | 20.4 |
|    15 |      2068 |        1931 |          72 |     3.48 |      2010 |      2015 |  81.6 |       0   |          62.2 | 20.3 |
|    16 |      2147 |        1989 |          94 |     4.38 |      2010 |      2013 |  81.6 |       0   |          62.2 | 20.9 |
|    17 |      2104 |        1929 |          82 |     3.9  |      2013 |      2013 |  84.1 |       0   |          62.4 | 21.3 |
|    18 |      2044 |        1907 |          80 |     3.91 |      2010 |      2015 |  83.7 |       0   |          61.6 | 20.5 |
|    19 |      2085 |        1959 |          75 |     3.6  |      2010 |      2013 |  81.6 |       0   |          62   | 21.5 |
|    20 |      2058 |        1945 |          78 |     3.79 |      2011 |      2014 |  84.4 |       0   |          61.8 | 20.8 |
|    21 |      2068 |        1944 |          80 |     3.87 |      2011 |      2014 |  82.9 |       0   |          62.1 | 19.9 |
|    22 |      2110 |        1990 |          64 |     3.03 |      2011 |      2014 |  83.1 |       0   |          60.9 | 21.9 |
|    23 |      1981 |        1887 |          73 |     3.69 |      2011 |      2014 |  84.7 |       0   |          61.3 | 20.1 |
|    24 |      2034 |        1930 |          66 |     3.24 |      2011 |      2014 |  84.1 |       0   |          61.6 | 19.8 |
|    25 |      2084 |        1937 |          63 |     3.02 |      2011 |      2014 |  82.7 |       0   |          60.8 | 19.2 |
|    26 |      2105 |        1972 |          78 |     3.71 |      2011 |      2014 |  84.8 |       0   |          61.3 | 20.7 |
|    27 |      2076 |        1972 |          78 |     3.76 |      2011 |      2014 |  84.8 |       0   |          62   | 20.7 |
|    28 |      2011 |        1894 |          78 |     3.88 |      2011 |      2014 |  82.9 |       0   |          61.5 | 20.6 |
|    29 |      2064 |        1933 |          83 |     4.02 |      2011 |      2014 |  82.8 |       0   |          61.2 | 20.8 |
|    30 |      2037 |        1930 |          82 |     4.03 |      2011 |      2014 |  84.3 |       0   |          62.4 | 20.9 |
|    31 |      2085 |        1977 |          69 |     3.31 |      2010 |      2014 |  81.8 |       0   |          61.3 | 19.8 |
|    32 |      2045 |        1828 |          94 |     4.6  |      2010 |      2010 |  77.7 |       0   |          61.2 | 20.9 |
|    33 |      2065 |        1865 |          89 |     4.31 |      2010 |      2010 |  78.6 |       0   |          60.2 | 21.5 |
|    34 |      2064 |        1833 |         107 |     5.18 |      2010 |      2010 |  80.6 |       0   |          60.2 | 21.7 |
|    35 |      2051 |        1847 |          90 |     4.39 |      2010 |      2010 |  78.8 |       0   |          60.8 | 20.7 |
|    36 |      2069 |        1956 |          89 |     4.3  |      2009 |      2011 |  79.7 |       0   |          60.2 | 21.3 |
|    37 |      2078 |        1889 |         123 |     5.92 |      2007 |      2011 |  77.7 |       0   |          61.2 | 19.7 |
|    38 |      2036 |        1912 |         116 |     5.7  |      2009 |      2011 |  81.5 |       0   |          60.5 | 18.8 |
|    39 |      2045 |        1869 |         118 |     5.77 |      2009 |      2011 |  78   |       0   |          61.1 | 19.8 |
|    40 |      2020 |        1875 |         103 |     5.1  |      2009 |      2011 |  81.2 |       0   |          59.9 | 19.5 |
|    41 |      1969 |        1850 |          81 |     4.11 |      2009 |      2011 |  80.5 |       0   |          61   | 20.3 |
|    42 |      2052 |        1867 |          97 |     4.73 |      2009 |      2011 |  76.7 |       0   |          61.3 | 19.4 |
|    43 |      2060 |        1906 |          88 |     4.27 |      2009 |      2011 |  81.4 |       0   |          60   | 19.2 |
|    44 |      1983 |        1844 |          96 |     4.84 |      2009 |      2011 |  78.8 |       0   |          60.6 | 19.7 |
|    45 |      2053 |        1904 |          92 |     4.48 |      2009 |      2011 |  79.6 |       0   |          61.1 | 18.5 |
|    46 |      2026 |        1890 |          94 |     4.64 |      2009 |      2011 |  82.4 |       0   |          60   | 20.8 |
|    47 |      2063 |        1916 |         109 |     5.28 |      2009 |      2011 |  79.4 |       0   |          60.9 | 19.8 |
|    48 |      2047 |        1871 |          94 |     4.59 |      2009 |      2011 |  79   |       0   |          61   | 20.9 |
|    49 |      2053 |        1776 |          92 |     4.48 |      2009 |      2011 |  80.1 |       0   |          61   | 20.6 |
|    50 |         1 |           1 |           1 |   100    |      2010 |      2010 | 100   |       0   |          59   |  0   |
|    54 |      2755 |        2540 |          72 |     2.61 |      2009 |      2017 |  91.3 |     100   |          62   | 31.4 |

## 10. Associação pacote × características (V de Cramér)
Referência: < 0,10 desprezível; 0,10–0,30 fraca; > 0,30 forte.
|                    |     V |
|:-------------------|------:|
| study_year         | 0.566 |
| pneumonia          | 0.044 |
| Projection         | 0.118 |
| Manufacturer_DICOM | 0.293 |
| MethodLabel        | 0.993 |
| PatientSex_DICOM   | 0.03  |
| faixa_etaria       | 0.062 |

- Pacientes com estudos em mais de um pacote: 18,055 (28.8%)

## 11. Pacotes necessários para 750 positivos (1000 sorteios, semente 2026)
- Pacotes: mediana 10, p90 11, máx 13
- GB: mediana 207, p90 228
- Positivos por pacote: média 78.4, mín 1, máx 123

## 12. Subconjunto candidato da D-025 — pacotes [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 54]

### 12.1 Contagens
- Download: 13 pacotes, 282.1 GB
- Estudos no recorte: 24,832; após remover rótulo 'exclude': 24,775 (removidos 57)
- Pacientes: 22,988
- Positivos, só 'pneumonia': **796** (3.2%)
- Positivos, incluindo 'atypical pneumonia': **812** (só atípica: 16)
- Portão G3 (≥ 750): atingido

- Estudos rotulados por médico fora dos pacotes escolhidos: 290 (positivos: 9); por pacote: 12: 290

### 12.2 Representatividade: candidato × restante × todos (sem 'exclude')
|                |   candidato |   restante |     todos |
|:---------------|------------:|-----------:|----------:|
| estudos        |    24775    |   77369    | 102144    |
| pacientes      |    22988    |   46803    |  62373    |
| positivos      |      796    |    3283    |   4079    |
| prevalência_%  |        3.21 |       4.24 |      3.99 |
| %PA            |       91.68 |      81.52 |     83.99 |
| %AP            |        1.29 |       3.62 |      3.06 |
| %AP_horizontal |        7.03 |      14.85 |     12.96 |
| %Philips       |       74.04 |      40.47 |     48.61 |
| %feminino      |       52.3  |      49.09 |     49.87 |
| idade_média    |       61.86 |      61.24 |     61.39 |
| %2007–2011     |        0.47 |      55.06 |     41.82 |
| %2012–2014     |       23.69 |      44.56 |     39.5  |
| %2015–2017     |       75.84 |       0.38 |     18.68 |
| %rótulo médico |      100    |       0.37 |     24.54 |

### 12.3 Positivos no candidato por projeção
| Projection    |   estudos |   positivos |   com_atipica |   prevalência_% |
|:--------------|----------:|------------:|--------------:|----------------:|
| AP            |       319 |          27 |            27 |            8.46 |
| AP_horizontal |      1742 |          98 |            98 |            5.63 |
| PA            |     22714 |         671 |           687 |            2.95 |

### 12.3 Positivos no candidato por fabricante
| Manufacturer_DICOM        |   estudos |   positivos |   com_atipica |   prevalência_% |
|:--------------------------|----------:|------------:|--------------:|----------------:|
| ImagingDynamicsCompanyLtd |      6432 |          80 |            82 |            1.24 |
| PhilipsMedicalSystems     |     18343 |         716 |           730 |            3.9  |

### 12.3 Positivos no candidato por ano
|   study_year |   estudos |   positivos |   com_atipica |   prevalência_% |
|-------------:|----------:|------------:|--------------:|----------------:|
|         2009 |       116 |           7 |             8 |            6.03 |
|         2014 |      5869 |         178 |           182 |            3.03 |
|         2015 |      6868 |         243 |           247 |            3.54 |
|         2016 |      6777 |         207 |           211 |            3.05 |
|         2017 |      5145 |         161 |           164 |            3.13 |

### 12.4 Parâmetros físicos no candidato, por fabricante (zeros tratados como ausentes)
|                                                             |   preenchido_% |   zero_% |   útil_% |   mediana |
|:------------------------------------------------------------|---------------:|---------:|---------:|----------:|
| ('ImagingDynamicsCompanyLtd', 'XRayTubeCurrent_DICOM')      |           99.8 |      0   |     99.8 |     320   |
| ('ImagingDynamicsCompanyLtd', 'Exposure_DICOM')             |           99.7 |      0   |     99.7 |       3   |
| ('ImagingDynamicsCompanyLtd', 'ExposureInuAs_DICOM')        |           99.7 |      0   |     99.7 |    3200   |
| ('ImagingDynamicsCompanyLtd', 'ExposureTime')               |           99.7 |      0   |     99.7 |      10   |
| ('ImagingDynamicsCompanyLtd', 'RelativeXRayExposure_DICOM') |           51.6 |      0.1 |     51.5 |      -1.2 |
| ('ImagingDynamicsCompanyLtd', 'WindowCenter_DICOM')         |           99.2 |      0   |     99.2 |    2189   |
| ('ImagingDynamicsCompanyLtd', 'WindowWidth_DICOM')          |           99.2 |      0   |     99.2 |    2363   |
| ('PhilipsMedicalSystems', 'XRayTubeCurrent_DICOM')          |           24.7 |     24.7 |      0   |     nan   |
| ('PhilipsMedicalSystems', 'Exposure_DICOM')                 |           69.6 |     24.7 |     44.9 |       2   |
| ('PhilipsMedicalSystems', 'ExposureInuAs_DICOM')            |           44.9 |      0   |     44.9 |    1500   |
| ('PhilipsMedicalSystems', 'ExposureTime')                   |           69.6 |     24.7 |     44.9 |       6   |
| ('PhilipsMedicalSystems', 'RelativeXRayExposure_DICOM')     |           99.3 |      0   |     99.3 |     353   |
| ('PhilipsMedicalSystems', 'WindowCenter_DICOM')             |           99.3 |      0   |     99.3 |    2047   |
| ('PhilipsMedicalSystems', 'WindowWidth_DICOM')              |           99.3 |      0   |     99.3 |    4095   |
