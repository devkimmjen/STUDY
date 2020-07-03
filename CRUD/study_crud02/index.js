const mysql = require('mysql');
const express = require('express');
var app = express();
const bodyparser = require('body-parser');

app.use(bodyparser.json());

var mysqlConnection = mysql.createConnection({
    host: 'localhost',
    user: 'root',
    password: 'a197346852',
    database: 'employeedb',
    multipleStatements: true
});

mysqlConnection.connect((err) => {
    if(!err)
        console.log('DB connection succeded.')
    else
        console.log('DB connection failed \n Error : ' + JSON.stringify(err, undefined, 2));
});

app.listen(3000, () => console.log('Express server is running at port no : 3000'));

// GET all employees
app.get('/employees', (req, res) => {
    mysqlConnection.query('SELECT * FROM Employee', (err, rows, fields) => {
        if(!err)
            // console.log(rows);
            // 화면에 표시
            res.send(rows);
        else
            console.log(err)
    })
});

// GET an employees
app.get('/employees/:id', (req, res) => {
    mysqlConnection.query('SELECT * FROM Employee WHERE = EmpID = ?',[req.params.id], (err, rows, fields) => {
        if(!err)
            // console.log(rows);
            // 화면에 표시
            res.send(rows);
        else
            console.log(err)
    })
});

// Delete an employees
app.delete('/employees/:id', (req, res) => {
    mysqlConnection.query('DELETE Employee WHERE = EmpID = ?',[req.params.id], (err, rows, fields) => {
        if(!err)
            // console.log(rows);
            // 화면에 표시
            res.send('Deleted successfully.');
        else
            console.log(err)
    })
});

// Update an employees
app.post('/employees', (req, res) => {
    let emp = req.body;
    var sql = "SET @EmpID = ?; SET @Name = ?; SET @EmpCode = ?; SET @Salary = ?; \
    CALL EmployeeAddOrEdit(@EmpID, @Name, @EmpCode, @Salary);";
    mysqlConnection.query(sql, [emp.EmpID, emp.Name, emp.EmpCode, emp.Salary], (err, rows, fields) => {
        if(!err)
            // console.log(rows);
            // 화면에 표시
            // rows.forEach(element => {
            //     if(element.constructor = Array)
            //     res.send('Inserted employee id : '+ element[0].EmpID);
            // });
            res.send('Updated successfully');
        else
            console.log(err)
    })
});