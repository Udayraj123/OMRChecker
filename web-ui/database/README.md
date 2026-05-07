# MinSU OMR Scanner Database

This folder contains the MySQL import file for Laragon/phpMyAdmin.

Database name: `minsu_omr_scanner`

Default connection used by the Next.js app:

```env
MYSQL_HOST=127.0.0.1
MYSQL_PORT=3306
MYSQL_USER=root
MYSQL_PASSWORD=
MYSQL_DATABASE=minsu_omr_scanner
```

Import `schema.sql` in phpMyAdmin, or download the browser-accessible copy from:

```txt
http://127.0.0.1:3000/database/minsu_omr_scanner_import.sql
```
