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


ALTER TABLE fnspid.`external` DROP COLUMN Lsa_summary;
ALTER TABLE fnspid.`external` DROP COLUMN Luhn_summary;
ALTER TABLE fnspid.`external` DROP COLUMN Textrank_summary;
ALTER TABLE fnspid.`external` DROP COLUMN row_id;
ALTER TABLE fnspid.`external` DROP COLUMN Lexrank_summary;



select * from fnspid.`external` e ;

CREATE view dates as SELECT *,
    SUBSTRING_INDEX(Date, ' ', 1) AS date_time,
    SUBSTRING_INDEX(SUBSTRING_INDEX(date, ' ', -2), ' ', 1) as time,
    SUBSTRING_INDEX(Date, ' ', -1) AS time_zone
FROM  
    `external`; 



select * from dates;

select distinct from time_zones;
select count(distinct time) from dates; 

select Article_title, Stock_symbol, url, date_time, time from dates;


drop VIEW Dates;



