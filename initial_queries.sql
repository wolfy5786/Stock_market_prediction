CREATE TABLE `fnspid`.`external` (
  `Date` DATETIME NULL,
  `Article_title` VARCHAR(45) NULL,
  `Stock_symbol` VARCHAR(45) NULL,
  `Url` VARCHAR(45) NULL,
  `Publisher` VARCHAR(45) NULL,
  `Author` VARCHAR(45) NULL,
  `Article` VARCHAR(45) NULL,
  `Lsa_summary` VARCHAR(45) NULL,
  `Luhn_summary` VARCHAR(45) NULL);

SHOW VARIABLES LIKE 'local_infile';


LOAD DATA LOCAL INFILE 'D:\\datafiles\\acedemics\\sem_7\\machine_learning\\north_eastern\\supervised\\project\\All_external.csv'
INTO TABLE fnspid.external
FIELDS TERMINATED BY ','  
ENCLOSED BY '"'  
LINES TERMINATED BY '\n'  
IGNORE 1 ROWS;


