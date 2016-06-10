

#ifndef _Q_IMAGE_WIDGET_H
#define _Q_IMAGE_WIDGET_H

#include <QLabel>


class QImageWidget : public QLabel
{
    Q_OBJECT

public:
    QImageWidget(QWidget* parent = nullptr);
    

public slots:
	bool loadFile(const QString &);
	void setImage(const QImage& image);


Q_SIGNALS:
	void closed();

protected:
	//virtual void paintEvent(QPaintEvent *);
	virtual void keyReleaseEvent(QKeyEvent *);

	virtual void closeEvent(QCloseEvent *event);
};

#endif	// _Q_IMAGE_WIDGET_H
