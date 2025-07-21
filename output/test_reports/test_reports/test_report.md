# Test Execution Report

**Generated**: 2025-07-21 12:33:00
**Status**: ❌ FAILED
**Exit Code**: 2

## Test Configuration

- **Test Mode**: Regular Tests
- **Verbose Output**: True
- **Parallel Execution**: False
- **Coverage Enabled**: True

## Test Statistics

- **Total Tests**: 350
- **Passed**: ✅ 0
- **Failed**: ❌ 0
- **Skipped**: ⏭️ 0
- **Errors**: 🚨 4
- **Warnings**: ⚠️ 0
- **Xfailed**: 0
- **Xpassed**: 0
- **Deselected**: 11

**Success Rate**: 0.0%
**Failure Rate**: 1.1%
**Execution Time**: 11.8 seconds

## Coverage Statistics

- **Total Coverage**: 5.6%
- **Covered Lines**: 1136
- **Missing Lines**: 13869
- **Total Statements**: 15005
- **Files Covered**: 81
- **Branch Coverage**: 0.0%
- **Line Coverage**: 0.0%

## Test Dependencies

- ✅ **pytest** (v8.4.0)
- ✅ **pytest-cov** (v6.1.1)
- ❌ **pytest-json-report** (vN/A)
- ❌ **pytest-xdist** (vN/A)
- ✅ **coverage** (v7.8.2)
- ❌ **mock** (vN/A)
- ✅ **psutil** (v7.0.0)

## Generated Reports

- **Xml Report**: `/home/trim/Documents/GitHub/GeneralizedNotationNotation/output/test_reports/test_reports/pytest_report.xml`
- **Markdown Report**: `/home/trim/Documents/GitHub/GeneralizedNotationNotation/output/test_reports/test_reports/test_report.md`
- **Coverage Html**: `/home/trim/Documents/GitHub/GeneralizedNotationNotation/output/test_reports/test_reports/coverage`
- **Coverage Json**: `/home/trim/Documents/GitHub/GeneralizedNotationNotation/output/test_reports/test_reports/test_coverage.json`

## Execution Details

- **Command**: `/home/trim/Documents/GitHub/GeneralizedNotationNotation/src/.venv/bin/python -m pytest --verbose --tb=short --junitxml=/home/trim/Documents/GitHub/GeneralizedNotationNotation/output/test_reports/test_reports/pytest_report.xml --maxfail=20 --durations=15 --disable-warnings -c/home/trim/Documents/GitHub/GeneralizedNotationNotation/pytest.ini --cov=src/gnn --cov=src/pipeline --cov=src/utils --cov-report=html:/home/trim/Documents/GitHub/GeneralizedNotationNotation/output/test_reports/test_reports/coverage --cov-report=json:/home/trim/Documents/GitHub/GeneralizedNotationNotation/output/test_reports/test_reports/test_coverage.json --cov-report=term-missing --cov-fail-under=0 --cov-config=/home/trim/Documents/GitHub/GeneralizedNotationNotation/.coveragerc --cov-branch -m not slow /home/trim/Documents/GitHub/GeneralizedNotationNotation/src/tests`
- **Working Directory**: `/home/trim/Documents/GitHub/GeneralizedNotationNotation`
- **Timeout**: 600 seconds

## Raw Output Preview

### Standard Output (last 50 lines)
```
src/gnn/parsers/markdown_serializer.py         69     62     40      0     6%   12-98
src/gnn/parsers/temporal_parser.py            264    236    104      0     8%   27-32, 35, 38-45, 49-134, 138-152, 156-221, 225-247, 251-261, 268-272, 275, 278-285, 289-352, 356-370, 374-439, 443-448, 452-459, 464, 471-484, 488-505, 509-517
src/gnn/parsers/protobuf_parser.py            234    209     92      0     8%   26-27, 30, 34-41, 45, 49-111, 121-140, 144-158, 162-214, 218-258, 262-291, 295-302, 307-315, 319-329, 334-349, 353-411, 415-424, 428-437, 441-457, 461-476
src/gnn/parsers/schema_parser.py              700    624    282      0     8%   26, 30-46, 50-105, 109-117, 120, 123-131, 135-146, 149-177, 180-189, 192-198, 204-207, 211-228, 232-287, 291-299, 302, 305-312, 315-354, 357-361, 364-375, 378-386, 392-396, 400-416, 420-476, 480-488, 491, 494-501, 504-581, 585, 591-595, 599-609, 613-624, 628-638, 642-698, 709-723, 727-746, 750-782, 786-813, 817-847, 851-858, 862-866, 870-879, 883-901, 905-925, 929-937, 941-949, 953-972, 976-985, 991-993, 996, 1000-1009, 1013-1071, 1074-1081, 1084-1119, 1122-1128, 1134-1136, 1139, 1143-1154, 1158-1216, 1219-1226, 1229-1264, 1267-1274, 1277-1283
src/gnn/parsers/xml_serializer.py             101     89     54      0     8%   15-163, 167-169, 178-180
src/gnn/parsers/python_parser.py              257    224    150      0     8%   30-33, 37-50, 55-71, 76-81, 85-99, 103-118, 122-187, 191-204, 208-229, 233-243, 247-267, 273-283, 287-289, 293-300, 305-310, 315-335, 340, 344-349, 353-357, 361-370, 376-403, 407-426, 430-448, 452-467, 471
src/gnn/parsers/xml_parser.py                 344    306    120      0     8%   30, 34-67, 71-109, 113-124, 130-164, 168-172, 177-188, 192-235, 240-255, 259-314, 319-329, 333-351, 355-366, 370-379, 383-398, 402-416, 424-434, 438-454, 458, 462-473, 478-530, 538, 544-556, 561-579, 583-591, 601-609, 619-625, 634, 638-649, 654-706
src/gnn/parsers/maxima_parser.py              149    131     56      0     9%   25-29, 33-42, 46-104, 108, 112-119, 123-196, 200-211, 215, 222-233, 237-250, 254-270, 274-294
src/gnn/parsers/scala_parser.py               261    228    114      0     9%   38-46, 58, 62-69, 74-107, 111-125, 129-194, 199-213, 217-240, 244-272, 276-308, 312-322, 326-331, 336-347, 352-365, 370-383, 388-401, 407-412, 417-438, 442-469, 473-474, 479-488, 493-507, 511-524, 528-536
src/gnn/parsers/asn1_serializer.py             96     85     28      0     9%   12-13, 17-117, 121-128, 132
src/gnn/parsers/isabelle_parser.py            148    130     54      0     9%   26-31, 35, 39-46, 51-127, 131-144, 148-213, 217-227, 232-238, 242-250, 255-256, 261-269, 273-294
src/gnn/parsers/protobuf_serializer.py         96     86     16      0     9%   13-147, 151-154, 163-176
src/gnn/parsers/lean_parser.py                176    154     70      0     9%   29-34, 38-51, 56-72, 76-90, 94-159, 163-183, 188-198, 202-210, 214-224, 229-251, 255-274, 278-296, 300-327, 331-343, 347
src/gnn/parsers/json_parser.py                180    160     34      0     9%   31, 35-57, 61-92, 96-107, 113-171, 175-214, 218-247, 251-273, 277-299, 303-316, 320-340, 344
src/gnn/parsers/grammar_parser.py             203    176     78      0    10%   29-32, 36-47, 51-109, 113-126, 130-146, 150-160, 165-183, 187-205, 209-236, 240-251, 257-269, 273, 281-286, 290-348, 352-362, 366-386, 391-402, 408-425, 429
src/gnn/parsers/functional_parser.py          126    109     42      0    10%   26-30, 33, 36-43, 47-113, 117-131, 135-200, 204-205, 209-216, 220, 226-234, 238-244
src/gnn/parsers/coq_parser.py                 167    144     60      0    10%   29-35, 39-52, 57-73, 77-90, 94-159, 164-183, 187-191, 195-203, 207-222, 226-244, 248-266, 270-281, 285-305, 309-320, 325-330, 334
src/gnn/parsers/binary_parser.py              134    114     54      0    11%   27, 30, 33-48, 52-60, 64-98, 102-103, 107-176, 181-198, 202-215, 220-229, 233-256, 260-270, 274-282
src/gnn/parsers/alloy_serializer.py            75     64     28      0    11%   12-13, 17-96, 101-105, 109-117
src/gnn/parsers/validators.py                 239    195    160      0    11%   40, 53-61, 65, 69, 73-78, 90, 102-122, 127-146, 155-170, 175-207, 211-247, 257-295, 300-330, 340-356, 366-387, 391-426, 437-506, 511-566, 571-617
src/gnn/parsers/maxima_serializer.py           53     43     30      0    12%   13-96, 100-102, 111-113
src/gnn/parsers/python_serializer.py           60     50     14      0    14%   13-100, 104-106, 115-117
src/gnn/parsers/scala_serializer.py            53     42     22      0    15%   13-89, 93-95, 104-106, 117-124
src/gnn/parsers/xsd_serializer.py              59     48     12      0    15%   13-92, 96-99, 108-121, 125-133
src/gnn/parsers/znotation_serializer.py        52     41     16      0    16%   13-89, 93-101, 105-107, 116-118
src/gnn/parsers/grammar_serializer.py          47     37     14      0    16%   13-89, 93-95, 104-106
src/gnn/parsers/functional_serializer.py       47     36     18      0    17%   13-84, 88-90, 99-101, 112-120
src/gnn/parsers/lean_serializer.py             47     36     12      0    19%   13-85, 89-91, 100-102, 113-121
src/gnn/parsers/temporal_serializer.py         77     58     16      0    20%   13-14, 18-21, 25-90, 94-96, 105-107, 118-126, 130-187, 191-199, 206, 213
src/gnn/parsers/coq_serializer.py              40     29      8      0    23%   13-76, 80-82, 91-93, 104-112
src/gnn/parsers/yaml_serializer.py             31     21     12      0    23%   6-7, 17-83, 87-105
src/gnn/parsers/isabelle_serializer.py         38     27      8      0    24%   13-74, 78-80, 89-91, 102-110
src/gnn/parsers/base_serializer.py             41     26      8      0    31%   10, 19-21, 25-28, 37-50, 54, 102-122, 126-134, 138-147
src/gnn/parsers/common.py                     381    196    124      0    37%   33-37, 41-48, 132-133, 137-143, 147, 160-161, 173-174, 185-186, 197-198, 209-210, 220-221, 231-232, 271, 275-278, 282-287, 291-294, 298-320, 324-347, 379, 385, 389-402, 413, 417, 421, 428, 432, 451, 455, 459-460, 464, 474-476, 497, 506, 530-541, 545-565, 569-594, 606-655, 659-670
src/gnn/parsers/converters.py                  21     11      2      0    43%   24, 41-49, 60-61, 74
src/gnn/parsers/binary_serializer.py           27     13      4      0    45%   17-58, 62-100, 104, 108-110, 119-121
src/gnn/types.py                              189     48     18      0    68%   19-22, 25-33, 50-53, 56-59, 139-140, 143, 172-173, 176, 179-180, 196-204, 207, 210-220
src/gnn/parsers/json_serializer.py              9      2      0      0    78%   13-62
src/gnn/parsers/schema_serializer.py            8      1      0      0    88%   13
---------------------------------------------------------------------------------------
TOTAL                                       15005  13869   5480      2     6%
Coverage HTML written to dir /home/trim/Documents/GitHub/GeneralizedNotationNotation/output/test_reports/test_reports/coverage
Coverage JSON written to file /home/trim/Documents/GitHub/GeneralizedNotationNotation/output/test_reports/test_reports/test_coverage.json
=========================== short test summary info ============================
ERROR src/tests/test_comprehensive_api.py - RecursionError: maximum recursion...
ERROR src/tests/test_export.py - RecursionError: maximum recursion depth exce...
ERROR src/tests/test_gnn_type_checker.py - RecursionError: maximum recursion ...
ERROR src/tests/test_parsers.py - RecursionError: maximum recursion depth exc...
!!!!!!!!!!!!!!!!!!! Interrupted: 4 errors during collection !!!!!!!!!!!!!!!!!!!!
================= 11 deselected, 1 warning, 4 errors in 11.76s =================
```

### Standard Error (last 20 lines)
```
/home/trim/Documents/GitHub/GeneralizedNotationNotation/src/.venv/lib/python3.10/site-packages/coverage/report_core.py:116: CoverageWarning: Couldn't parse '/home/trim/Documents/GitHub/GeneralizedNotationNotation/src/gnn/parsers/unified_parser.py': maximum recursion depth exceeded in comparison (couldnt-parse)
  coverage._warn(msg, slug="couldnt-parse")
```
