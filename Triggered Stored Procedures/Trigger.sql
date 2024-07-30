DELIMITER //

CREATE TRIGGER after_insert_tables
AFTER INSERT ON tables
FOR EACH ROW
BEGIN
    -- Create columns for the newly created table
    DELCLARE coumn_name VARCHAR(100);
    DECLARE data_type VARCHAR(50);
    DECLARE column_cursor CURSOR FOR 
    SELECT table_columns 
    WHERE table_id = NEW.id;

    OPEN column_cursor;

    column_loop: LOOP 
        FETCH column_cursor INTO column_name, data_type;
        IF done THEN 
            LEAVE column_loop;
        END IF;

        SET @add_column_sql = CONCAT('ALTER TABLE', NEW.table_name, ' ADD COLUMN ', column_name, ' ', data_type);
        PREPARE stmt FROM @add_column_sql;
        EXECUTE stmt;
        DEALLOCATE PREPARE stmt;
    END LOOP;

    CLOSE column_cursor;
END //

DELIMITER ;