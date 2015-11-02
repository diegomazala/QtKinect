/****************************************************************************
** Meta object code from reading C++ file 'QKinectReader.h'
**
** Created by: The Qt Meta Object Compiler version 67 (Qt 5.5.0)
**
** WARNING! All changes made in this file will be lost!
*****************************************************************************/

#include "../../../src/QKinectReader.h"
#include <QtCore/qbytearray.h>
#include <QtCore/qmetatype.h>
#if !defined(Q_MOC_OUTPUT_REVISION)
#error "The header file 'QKinectReader.h' doesn't include <QObject>."
#elif Q_MOC_OUTPUT_REVISION != 67
#error "This file was generated using the moc from 5.5.0. It"
#error "cannot be used with the include files from this version of Qt."
#error "(The moc has changed too much.)"
#endif

QT_BEGIN_MOC_NAMESPACE
struct qt_meta_stringdata_QKinectReader_t {
    QByteArrayData data[8];
    char stringdata0[75];
};
#define QT_MOC_LITERAL(idx, ofs, len) \
    Q_STATIC_BYTE_ARRAY_DATA_HEADER_INITIALIZER_WITH_OFFSET(len, \
    qptrdiff(offsetof(qt_meta_stringdata_QKinectReader_t, stringdata0) + ofs \
        - idx * sizeof(QByteArrayData)) \
    )
static const qt_meta_stringdata_QKinectReader_t qt_meta_stringdata_QKinectReader = {
    {
QT_MOC_LITERAL(0, 0, 13), // "QKinectReader"
QT_MOC_LITERAL(1, 14, 10), // "colorImage"
QT_MOC_LITERAL(2, 25, 0), // ""
QT_MOC_LITERAL(3, 26, 5), // "image"
QT_MOC_LITERAL(4, 32, 10), // "depthImage"
QT_MOC_LITERAL(5, 43, 13), // "infraredImage"
QT_MOC_LITERAL(6, 57, 12), // "frameUpdated"
QT_MOC_LITERAL(7, 70, 4) // "stop"

    },
    "QKinectReader\0colorImage\0\0image\0"
    "depthImage\0infraredImage\0frameUpdated\0"
    "stop"
};
#undef QT_MOC_LITERAL

static const uint qt_meta_data_QKinectReader[] = {

 // content:
       7,       // revision
       0,       // classname
       0,    0, // classinfo
       5,   14, // methods
       0,    0, // properties
       0,    0, // enums/sets
       0,    0, // constructors
       0,       // flags
       4,       // signalCount

 // signals: name, argc, parameters, tag, flags
       1,    1,   39,    2, 0x06 /* Public */,
       4,    1,   42,    2, 0x06 /* Public */,
       5,    1,   45,    2, 0x06 /* Public */,
       6,    0,   48,    2, 0x06 /* Public */,

 // slots: name, argc, parameters, tag, flags
       7,    0,   49,    2, 0x0a /* Public */,

 // signals: parameters
    QMetaType::Void, QMetaType::QImage,    3,
    QMetaType::Void, QMetaType::QImage,    3,
    QMetaType::Void, QMetaType::QImage,    3,
    QMetaType::Void,

 // slots: parameters
    QMetaType::Void,

       0        // eod
};

void QKinectReader::qt_static_metacall(QObject *_o, QMetaObject::Call _c, int _id, void **_a)
{
    if (_c == QMetaObject::InvokeMetaMethod) {
        QKinectReader *_t = static_cast<QKinectReader *>(_o);
        Q_UNUSED(_t)
        switch (_id) {
        case 0: _t->colorImage((*reinterpret_cast< const QImage(*)>(_a[1]))); break;
        case 1: _t->depthImage((*reinterpret_cast< const QImage(*)>(_a[1]))); break;
        case 2: _t->infraredImage((*reinterpret_cast< const QImage(*)>(_a[1]))); break;
        case 3: _t->frameUpdated(); break;
        case 4: _t->stop(); break;
        default: ;
        }
    } else if (_c == QMetaObject::IndexOfMethod) {
        int *result = reinterpret_cast<int *>(_a[0]);
        void **func = reinterpret_cast<void **>(_a[1]);
        {
            typedef void (QKinectReader::*_t)(const QImage & );
            if (*reinterpret_cast<_t *>(func) == static_cast<_t>(&QKinectReader::colorImage)) {
                *result = 0;
            }
        }
        {
            typedef void (QKinectReader::*_t)(const QImage & );
            if (*reinterpret_cast<_t *>(func) == static_cast<_t>(&QKinectReader::depthImage)) {
                *result = 1;
            }
        }
        {
            typedef void (QKinectReader::*_t)(const QImage & );
            if (*reinterpret_cast<_t *>(func) == static_cast<_t>(&QKinectReader::infraredImage)) {
                *result = 2;
            }
        }
        {
            typedef void (QKinectReader::*_t)();
            if (*reinterpret_cast<_t *>(func) == static_cast<_t>(&QKinectReader::frameUpdated)) {
                *result = 3;
            }
        }
    }
}

const QMetaObject QKinectReader::staticMetaObject = {
    { &QThread::staticMetaObject, qt_meta_stringdata_QKinectReader.data,
      qt_meta_data_QKinectReader,  qt_static_metacall, Q_NULLPTR, Q_NULLPTR}
};


const QMetaObject *QKinectReader::metaObject() const
{
    return QObject::d_ptr->metaObject ? QObject::d_ptr->dynamicMetaObject() : &staticMetaObject;
}

void *QKinectReader::qt_metacast(const char *_clname)
{
    if (!_clname) return Q_NULLPTR;
    if (!strcmp(_clname, qt_meta_stringdata_QKinectReader.stringdata0))
        return static_cast<void*>(const_cast< QKinectReader*>(this));
    return QThread::qt_metacast(_clname);
}

int QKinectReader::qt_metacall(QMetaObject::Call _c, int _id, void **_a)
{
    _id = QThread::qt_metacall(_c, _id, _a);
    if (_id < 0)
        return _id;
    if (_c == QMetaObject::InvokeMetaMethod) {
        if (_id < 5)
            qt_static_metacall(this, _c, _id, _a);
        _id -= 5;
    } else if (_c == QMetaObject::RegisterMethodArgumentMetaType) {
        if (_id < 5)
            *reinterpret_cast<int*>(_a[0]) = -1;
        _id -= 5;
    }
    return _id;
}

// SIGNAL 0
void QKinectReader::colorImage(const QImage & _t1)
{
    void *_a[] = { Q_NULLPTR, const_cast<void*>(reinterpret_cast<const void*>(&_t1)) };
    QMetaObject::activate(this, &staticMetaObject, 0, _a);
}

// SIGNAL 1
void QKinectReader::depthImage(const QImage & _t1)
{
    void *_a[] = { Q_NULLPTR, const_cast<void*>(reinterpret_cast<const void*>(&_t1)) };
    QMetaObject::activate(this, &staticMetaObject, 1, _a);
}

// SIGNAL 2
void QKinectReader::infraredImage(const QImage & _t1)
{
    void *_a[] = { Q_NULLPTR, const_cast<void*>(reinterpret_cast<const void*>(&_t1)) };
    QMetaObject::activate(this, &staticMetaObject, 2, _a);
}

// SIGNAL 3
void QKinectReader::frameUpdated()
{
    QMetaObject::activate(this, &staticMetaObject, 3, Q_NULLPTR);
}
QT_END_MOC_NAMESPACE
