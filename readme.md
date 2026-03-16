## Datasett

Rådataene som brukes i dette prosjektet er ikke inkludert i repositoryet. 
Filnavnene følger en fast struktur som beskriver både hvilken måler dataene kommer fra og hvilken type måledata filen inneholder.

### Navnestruktur

Filnavnene er på formen:

meter_id_datatype.csv

- **meter_id** beskriver hvilken måler dataene kommer fra.
- **sampletime** hyppighet på samples

### sampletime

Følgende datatyper brukes i filnavnene:

- `_time` 
- `_dag` 


### Plassering

CSV-filene forventes å ligge i prosjektets `data/`-mappe før skriptene kjøres.