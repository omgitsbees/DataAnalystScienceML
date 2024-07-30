DELIMITER //

CREATE Procedure CreateUserTable(
    IN p_user_id INT,
    IN p_table_name VARCHAR(100)
)

BEGIN
    DECLARE table_exists INT;

    -- Check if table already exists
    SELECT COUNT(*)