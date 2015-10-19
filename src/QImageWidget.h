

#ifndef QImageWidget_H
#define QImageWidget_H

#include <QLabel>


class QImageWidget : public QLabel
{
    Q_OBJECT

public:
    QImageWidget(QWidget* parent = nullptr);
    

public slots:
	bool loadFile(const QString &);
	void setImage(const QImage& image);

protected:
	//virtual void paintEvent(QPaintEvent *);
	virtual void keyReleaseEvent(QKeyEvent *);
};

#endif
