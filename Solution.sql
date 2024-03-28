-- Q175
SELECT
    P.firstName,
    P.lastName,
    A.city,
    A.state
FROM
    Person P
    LEFT JOIN Address A ON P.personId = A.personId;

-- Q176
SELECT
    MAX(salary) AS SecondHighestSalary
FROM
    employee
WHERE
    salary < (
        SELECT
            MAX(salary)
        FROM
            employee
    )